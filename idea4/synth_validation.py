"""
================================================================================
합성 데이터 정답 기반 검증 (Synthetic Ground-Truth Validation)
================================================================================
흐름
  [1] 콘크리트 실제 변수 범위 기반 합성 데이터 생성 (정답 G개 중심 포함)
  [2] 기존 파이프라인 그대로 적용: MRS-PRIM 박스 생성 -> 경험적 Jaccard ->
      average-linkage -> N_Memb x K 완전탐색으로 K* 확정
  [3] K*에서 나온 효과적 클러스터들을 정답(G개 중심)과 대조해 채점
        i)  개수 일치: N_eff == G ?
        ii) 위치 일치: 각 클러스터가 실제 정답 영역 근처 관측치와 얼마나
            일치하는지 (그리디 매칭 + Precision/Recall/F1)
  [4] 난이도(G, 중심간거리, noise, sigma)를 조합해 반복하고 결과표 제시
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

FEAT_NAMES = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
              'CoarseAgg', 'FineAgg', 'Age']
CONCRETE_RANGES_DEFAULT = {
    'Cement': (102.0, 540.0), 'Slag': (0.0, 359.4), 'FlyAsh': (0.0, 200.1),
    'Water': (121.8, 247.0), 'Superplast': (0.0, 32.2),
    'CoarseAgg': (801.0, 1145.0), 'FineAgg': (594.0, 992.6), 'Age': (1.0, 365.0),
}

# --- 파이프라인 하이퍼파라미터 (원논문 방식과 동일 계열) ---
ALPHA_PEEL = 0.05
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20
S_RATIO_LOW, S_RATIO_HIGH = 0.5, 0.75   # subspace 크기 = P의 50~75%
T_PER_SIZE = 300
MATCH_RADIUS_RATIO = 0.20   # '정답 영역 근처 관측치'로 인정할 정규화 반경


# ============================================= [1] 합성 데이터 생성
def load_real_ranges(csv_path):
    if not os.path.exists(csv_path):
        return CONCRETE_RANGES_DEFAULT
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    return {name: (float(X[:, i].min()), float(X[:, i].max()))
            for i, name in enumerate(FEAT_NAMES)}


def sample_centers(ranges, G, min_sep_ratio, seed):
    rng = np.random.default_rng(seed)
    P = len(ranges)
    lo = np.array([ranges[n][0] for n in FEAT_NAMES[:P]])
    hi = np.array([ranges[n][1] for n in FEAT_NAMES[:P]])
    centers, attempts = [], 0
    while len(centers) < G and attempts < 5000:
        cand = rng.uniform(lo, hi)
        if all(np.sqrt(np.mean(((cand - c) / (hi - lo)) ** 2)) >= min_sep_ratio
              for c in centers):
            centers.append(cand)
        attempts += 1
    if len(centers) < G:
        raise RuntimeError(f'min_sep_ratio={min_sep_ratio} 로 {G}개 중심 배치 실패')
    return np.array(centers)


def generate_synthetic_data(ranges, N, G, A, sigma_ratio, noise_std,
                            baseline, min_sep_ratio, seed):
    rng = np.random.default_rng(seed)
    P = len(ranges)
    feat_names = FEAT_NAMES[:P]
    lo = np.array([ranges[n][0] for n in feat_names])
    hi = np.array([ranges[n][1] for n in feat_names])
    span = hi - lo

    centers = sample_centers(ranges, G, min_sep_ratio, seed)
    X = rng.uniform(lo, hi, size=(N, P))

    D = np.full(N, baseline)
    dist_to_centers = np.zeros((N, G))
    for g in range(G):
        d_norm = np.sqrt(np.mean(((X - centers[g]) / span) ** 2, axis=1))
        dist_to_centers[:, g] = d_norm
        D += A[g] * np.exp(-(d_norm ** 2) / (2 * sigma_ratio ** 2))
    D += rng.normal(0, noise_std, size=N)
    D = np.clip(D, 0.0, 1.0)

    # 정답 관측치 집합: 각 중심의 정규화 거리가 MATCH_RADIUS_RATIO 이내인 관측치들
    truth_sets = [set(np.where(dist_to_centers[:, g] <= MATCH_RADIUS_RATIO)[0].tolist())
                 for g in range(G)]

    return dict(X=X, D=D, centers=centers, span=span, truth_sets=truth_sets,
               feat_names=feat_names)


# ============================================= [2] 기존 파이프라인 (MRS-PRIM + 클러스터링)
def peel_trajectory(X, D, S, min_support, rng):
    P = X.shape[1]
    idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < min_support:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
            for keep in (idx[xp > lo_q], idx[xp < hi_q]):
                if min_support <= len(keep) < len(idx):
                    obj = D[keep].mean()
                    if obj > cand_obj:
                        cand_obj, cand_keep = obj, keep
        if cand_keep is None:
            break
        idx = cand_keep
        if cand_obj > best_obj:
            best_obj, best_idx = cand_obj, idx.copy()
    return best_idx, best_obj


def build_boxes(X, D, min_support, seed):
    rng = np.random.default_rng(seed)
    P = X.shape[1]
    S_opts = tuple(sorted(set(int(round(P * r)) for r in
                              np.linspace(S_RATIO_LOW, S_RATIO_HIGH, 3))))
    boxes = []
    for S in S_opts:
        S = max(2, min(S, P - 1))
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, min_support, rng)
            if len(idx) >= min_support:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
    return boxes


def build_linkage(boxes):
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)
    sim = inter / (sup[:, None] + sup[None, :] - inter)
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average'), sets, M


def full_grid_search(Z, sets, M):
    all_candidates = []
    for n_memb in N_MEMB_OPTIONS:
        best_for_this = None
        for K in range(max(int(M * 0.05), 2), int(M * 0.5) + 1, K_GRID_STEP):
            lab = fcluster(Z, K, criterion='maxclust')
            gams, Meff, neff = [], 0, 0
            for k in np.unique(lab):
                mem = np.where(lab == k)[0]
                if len(mem) < n_memb:
                    continue
                I = set.intersection(*[sets[m] for m in mem])
                U = set.union(*[sets[m] for m in mem])
                gams.append(len(I) / len(U) if U else 0)
                Meff += len(mem); neff += 1
            if neff and Meff / M >= RHO_LIMIT:
                ecc = np.mean(gams)
                if best_for_this is None or ecc > best_for_this[2]:
                    best_for_this = (K, neff, ecc, Meff / M)
        if best_for_this is not None:
            K_star, N_eff, ecc, rho = best_for_this
            all_candidates.append((n_memb, K_star, N_eff, ecc, rho))
    if not all_candidates:
        return None
    return max(all_candidates, key=lambda c: c[3])


# ============================================= [3] 채점
def score_against_truth(boxes, lab, n_memb_star, truth_sets, N_total):
    """효과적 클러스터(관측치 union)와 정답 영역(truth_sets)을 그리디 매칭 후 F1 계산"""
    eff_clusters = []
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        if len(mem) < n_memb_star:
            continue
        U = set()
        for m in mem:
            U.update(boxes[m]['idx'].tolist())
        eff_clusters.append(U)

    G = len(truth_sets)
    N_eff = len(eff_clusters)

    # 그리디 매칭: 모든 (클러스터, 정답) F1 계산 후 높은 순으로 1:1 매칭
    pairs = []
    for ci, C in enumerate(eff_clusters):
        for gi, T in enumerate(truth_sets):
            inter = len(C & T)
            prec = inter / len(C) if C else 0.0
            rec = inter / len(T) if T else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pairs.append((f1, prec, rec, ci, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_c, matched_g, results = set(), set(), []
    for f1, prec, rec, ci, gi in pairs:
        if ci in matched_c or gi in matched_g:
            continue
        matched_c.add(ci); matched_g.add(gi)
        results.append((gi, ci, f1, prec, rec))

    results.sort(key=lambda r: r[0])
    mean_f1 = np.mean([r[2] for r in results]) if results else 0.0
    return dict(N_eff=N_eff, G=G, count_match=(N_eff == G),
               matches=results, mean_f1=mean_f1,
               unmatched_truth=G - len(results))


# ============================================= 실행: 시나리오 하나 검증
def run_scenario(ranges, label, N=1000, G=3, A=None, sigma_ratio=0.15,
                 noise_std=0.05, min_sep_ratio=0.15, min_support_ratio=0.10,
                 seed=0, verbose=True):
    if A is None:
        A = [0.7] * G
    data = generate_synthetic_data(ranges, N, G, A, sigma_ratio, noise_std,
                                   baseline=0.05, min_sep_ratio=min_sep_ratio,
                                   seed=seed)
    min_support = max(20, int(N * min_support_ratio))
    boxes = build_boxes(data['X'], data['D'], min_support, seed)
    if len(boxes) < 5:
        return dict(label=label, error='박스 생성 실패(min_support 과도)')

    Z, sets, M = build_linkage(boxes)
    final = full_grid_search(Z, sets, M)
    if final is None:
        return dict(label=label, error='rho_Limit 만족 K 없음')
    n_memb_star, K_star, N_eff, ecc, rho = final
    lab = fcluster(Z, K_star, criterion='maxclust')

    score = score_against_truth(boxes, lab, n_memb_star, data['truth_sets'], N)

    result = dict(label=label, G=G, N_eff=score['N_eff'],
                  count_match=score['count_match'], mean_f1=score['mean_f1'],
                  unmatched_truth=score['unmatched_truth'],
                  K_star=K_star, ECC=ecc, n_boxes=len(boxes))
    if verbose:
        print(f'  [{label}] G={G} -> N_eff={score["N_eff"]} '
              f'(개수일치={"O" if score["count_match"] else "X"}), '
              f'평균F1={score["mean_f1"]:.3f}, 매칭실패={score["unmatched_truth"]}개, '
              f'K*={K_star}, ECC={ecc:.3f}, 박스={len(boxes)}개')
    return result


# ============================================= 난이도별 반복 실행
def main():
    ranges = load_real_ranges(CSV_PATH)
    print(f'변수 범위 로드 완료 ({"실제 CSV" if os.path.exists(CSV_PATH) else "기본값"})\n')

    print('=' * 78)
    print('  난이도별 정답 복원 검증')
    print('=' * 78)

    results = []

    print('\n[A] 정답 개수(G) 변화 (거리/noise 고정)')
    for G in [2, 3, 5]:
        r = run_scenario(ranges, f'G={G}', N=1200, G=G,
                         A=[0.7]*G, sigma_ratio=0.15, noise_std=0.05,
                         min_sep_ratio=0.20, seed=1)
        results.append(r)

    print('\n[B] 중심 간 거리(min_sep_ratio) 변화 (G=3 고정)')
    for sep in [0.10, 0.20, 0.35]:
        r = run_scenario(ranges, f'sep={sep}', N=1200, G=3,
                         A=[0.7]*3, sigma_ratio=0.15, noise_std=0.05,
                         min_sep_ratio=sep, seed=2)
        results.append(r)

    print('\n[C] noise 수준 변화 (G=3 고정)')
    for noise in [0.02, 0.05, 0.10]:
        r = run_scenario(ranges, f'noise={noise}', N=1200, G=3,
                         A=[0.7]*3, sigma_ratio=0.15, noise_std=noise,
                         min_sep_ratio=0.20, seed=3)
        results.append(r)

    print('\n[D] 상대적 세기(A) 불균형 (G=3, 약한 영역 포함)')
    for A in ([0.7, 0.7, 0.7], [0.9, 0.5, 0.2]):
        r = run_scenario(ranges, f'A={A}', N=1200, G=3,
                         A=A, sigma_ratio=0.15, noise_std=0.05,
                         min_sep_ratio=0.20, seed=4)
        results.append(r)

    print('\n' + '=' * 78)
    print('  종합 요약')
    print('=' * 78)
    valid = [r for r in results if 'error' not in r]
    if valid:
        n_count_ok = sum(r['count_match'] for r in valid)
        print(f'  개수 일치(N_eff==G): {n_count_ok}/{len(valid)}개 시나리오')
        print(f'  평균 F1(전체 시나리오 평균): {np.mean([r["mean_f1"] for r in valid]):.3f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
