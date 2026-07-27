"""
================================================================================
아이디어 6번 [1/2] — rho^Limit 민감도 분석
================================================================================
원논문 식(12)는 K* = argmax_K ECC  s.t. rho_eff >= rho^Limit 로 정의되며,
rho^Limit=0.6 이 고정값으로 쓰였다. 이 하한을 0.50~0.70 구간에서 바꿔가며
K*, N_eff, ECC 가 얼마나 민감하게 반응하는지 관찰한다.

박스 생성(MRS-PRIM)과 유사도(경험적 Jaccard)는 원논문 표준 설정 그대로
1회만 계산하고, rho^Limit x N_Memb 조합만 바꿔가며 재사용한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_concrete.npy')

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]   # 민감도 분석 대상
K_GRID_STEP = 25


# ============================================= desirability (Koo Table1 NTB)
def make_desirability(y_all):
    lower, upper = float(y_all.min()), float(y_all.max())
    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= 60.0)
        d[m1] = (y[m1] - lower) / (60.0 - lower)
        m2 = (y > 60.0) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - 60.0)
        return np.clip(d, 0.0, 1.0)
    return d_func


# ============================================= MRS-PRIM 박스 생성 (원논문 표준)
def peel_trajectory(X, D, S, rng):
    P = X.shape[1]
    idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < MIN_SUPPORT:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
            for keep in (idx[xp > lo_q], idx[xp < hi_q]):
                if MIN_SUPPORT <= len(keep) < len(idx):
                    obj = D[keep].mean()
                    if obj > cand_obj:
                        cand_obj, cand_keep = obj, keep
        if cand_keep is None:
            break
        idx = cand_keep
        if cand_obj > best_obj:
            best_obj, best_idx = cand_obj, idx.copy()
    return best_idx, best_obj


def build_boxes(X, D):
    rng = np.random.default_rng(SEED_PRIM)
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 300 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 클러스터링 (1회 계산 후 재사용)
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


def precompute_per_K(Z, sets, M, k_grid):
    """K별 클러스터 결과(멤버수, Gamma)를 미리 계산해 재사용 (rho_Limit 스윕 시 중복 계산 방지)"""
    per_K = {}
    for K in k_grid:
        lab = fcluster(Z, K, criterion='maxclust')
        entries = []
        for k in np.unique(lab):
            mem = np.where(lab == k)[0]
            I = set.intersection(*[sets[m] for m in mem])
            U = set.union(*[sets[m] for m in mem])
            entries.append((len(mem), len(I) / len(U) if U else 0.0))
        per_K[K] = entries
    return per_K


def find_kstar_original(per_K, M, n_memb, rho_limit):
    """원논문 식(12): argmax ECC s.t. rho_eff >= rho_Limit"""
    best = None
    for K, entries in per_K.items():
        eff = [(n, g) for n, g in entries if n >= n_memb]
        if not eff:
            continue
        N_eff = len(eff)
        rho = sum(n for n, _ in eff) / M
        if rho < rho_limit:
            continue
        ecc = np.mean([g for _, g in eff])
        if best is None or ecc > best[1]:
            best = (K, ecc, N_eff, rho)
    return best   # (K*, ECC, N_eff, rho_eff) 또는 None


def get_top_clusters_obs(Z, boxes, K, n_memb, topN=3):
    """특정 K에서 효과적 클러스터를 멤버수 내림차순 정렬 후, 상위 topN개의
    관측치 합집합(observation union set)을 반환."""
    lab = fcluster(Z, K, criterion='maxclust')
    clusters = []
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        if len(mem) < n_memb:
            continue
        obs_union = set()
        for m in mem:
            obs_union.update(boxes[m]['idx'].tolist())
        clusters.append((len(mem), obs_union))
    clusters.sort(key=lambda c: -c[0])
    return clusters[:topN]


def jaccard_sets(A, B):
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B) if (A or B) else 0.0


# ============================================= 실행
def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    d_func = make_desirability(y)
    D = d_func(y)
    print(f'데이터 {len(y)}행\n')

    print('MRS-PRIM 박스 생성 (원논문 표준 설정)')
    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드\n')
    else:
        boxes = build_boxes(X, D)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'  박스 {len(boxes)}개 생성 및 저장\n')

    print('클러스터링(경험적 Jaccard -> average-linkage) 1회 계산 ...')
    Z, sets, M = build_linkage(boxes)
    k_grid = range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP)
    per_K = precompute_per_K(Z, sets, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid.start}~{k_grid.stop-1} (step {K_GRID_STEP})\n')

    print('=' * 90)
    print('  rho^Limit x N_Memb 별 K* 민감도 분석')
    print('=' * 90)
    print(f'  {"rho_Limit":>10} {"N_Memb":>7} {"K*":>6} {"N_eff":>6} '
          f'{"ECC":>8} {"rho_eff":>8}')

    summary = {}   # rho_limit -> (best K* across N_Memb 조합, 그때의 ECC)
    for rho_limit in RHO_LIMIT_GRID:
        rows = []
        for n_memb in N_MEMB_OPTIONS:
            res = find_kstar_original(per_K, M, n_memb, rho_limit)
            if res is None:
                print(f'  {rho_limit:>10.2f} {n_memb:>7}   (조건 만족 K 없음)')
                continue
            K_s, ecc, neff, rho = res
            rows.append((n_memb, K_s, neff, ecc, rho))
            print(f'  {rho_limit:>10.2f} {n_memb:>7} {K_s:>6} {neff:>6} '
                  f'{ecc:>8.4f} {rho:>8.3f}')
        if rows:
            best_row = max(rows, key=lambda r: r[3])
            summary[rho_limit] = best_row
        print('  ' + '-' * 86)

    print('\n' + '=' * 90)
    print('  요약: rho^Limit 별 "전체 최적"(ECC 최댓값) 조합')
    print('=' * 90)
    print(f'  {"rho_Limit":>10} {"N_Memb*":>8} {"K*":>6} {"N_eff":>6} {"ECC":>8}')
    prev_K = None
    for rho_limit, (n_memb, K_s, neff, ecc, rho) in summary.items():
        shift = ''
        if prev_K is not None:
            shift = f'  (이전 대비 K* {"증가" if K_s>prev_K else ("감소" if K_s<prev_K else "동일")})'
        print(f'  {rho_limit:>10.2f} {n_memb:>8} {K_s:>6} {neff:>6} {ecc:>8.4f}{shift}')
        prev_K = K_s

    # ---------- 핵심 클러스터 지속성 확인 ----------
    print('\n' + '=' * 90)
    print('  핵심 클러스터 지속성: K*는 달라져도 주요 클러스터가 유지되는가')
    print('=' * 90)
    print('  (기준: rho_Limit=0.60(원논문 값)에서의 상위 클러스터와, 다른 rho_Limit에서')
    print('   나온 상위 클러스터를 관측치 집합 Jaccard 유사도로 비교. N_Memb=5 고정)')

    N_MEMB_FIXED = 5
    baseline_rho = 0.60
    baseline_K = summary[baseline_rho][1]
    baseline_top = get_top_clusters_obs(Z, boxes, baseline_K, N_MEMB_FIXED, topN=3)
    print(f'\n  기준(rho_Limit=0.60, K*={baseline_K}) 상위 클러스터 크기: '
          f'{[len(c[1]) for c in baseline_top]} (관측치 수)')

    print(f'\n  {"rho_Limit":>10} {"K*":>6} {"rank1 Jaccard":>14} '
          f'{"rank2 Jaccard":>14} {"rank3 Jaccard":>14}')
    for rho_limit in RHO_LIMIT_GRID:
        if rho_limit not in summary:
            continue
        K_s = summary[rho_limit][1]
        top_clusters = get_top_clusters_obs(Z, boxes, K_s, N_MEMB_FIXED, topN=3)
        # 기준 클러스터 각각과 최적 매칭(가장 높은 Jaccard)을 찾아 비교
        jvals = []
        for _, base_obs in baseline_top:
            best_j = max((jaccard_sets(base_obs, obs) for _, obs in top_clusters),
                        default=0.0)
            jvals.append(best_j)
        jstr = [f'{j:.3f}' for j in jvals]
        tag = '  <-- 기준' if rho_limit == baseline_rho else ''
        print(f'  {rho_limit:>10.2f} {K_s:>6} {jstr[0]:>14} {jstr[1]:>14} '
              f'{jstr[2]:>14}{tag}')

    print('\n  해석: Jaccard가 1.0에 가까우면 그 순위의 핵심 클러스터가 rho_Limit')
    print('        설정과 무관하게 동일한 관측치 집합(=같은 운전영역)을 계속')
    print('        가리킨다는 뜻. 낮으면 rho_Limit 설정에 따라 발견되는 핵심')
    print('        영역 자체가 바뀐다는 뜻.')

    print('\n완료.')


if __name__ == '__main__':
    main()
