"""
================================================================================
아이디어 6번 [2/2] — ECC x rho_eff 통합 목적함수
================================================================================
기존(원논문 식12): K* = argmax_K ECC(K)   s.t.  rho_eff(K) >= rho^Limit
  -> rho^Limit 이라는 임의의 문턱값에 K*가 민감하게 반응함을 [1/2]에서 확인.

이번 개선: K* = argmax_K [ ECC(K) x rho_eff(K) ]
  -> 커버리지를 '제약'에서 '목적의 일부'로 격상. 조밀도만 밀어붙이던 구조에서
     벗어나, 조밀도와 커버리지가 동시에 높은 지점에서 자연스러운 내부
     최적점이 형성되는지 확인한다. 임의의 문턱값(rho^Limit)이 필요 없어진다.

비교: 기존 방식(rho^Limit=0.60, 원논문 값)과 신규 방식의 K*, ECC, rho_eff를
      나란히 제시하고, 신규 방식이 고른 K*에서도 핵심 클러스터(1~3위)가
      기존 방식의 핵심 클러스터와 얼마나 일치하는지(Jaccard) 확인한다.
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
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60   # 기존 방식과의 비교 기준(원논문 값)


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


def eval_K(entries, M, n_memb):
    """주어진 K의 클러스터 결과에서 N_Memb 기준 효과적 클러스터를 골라 ECC, rho_eff 계산"""
    eff = [(n, g) for n, g in entries if n >= n_memb]
    if not eff:
        return None
    N_eff = len(eff)
    rho = sum(n for n, _ in eff) / M
    ecc = np.mean([g for _, g in eff])
    return ecc, rho, N_eff


def find_kstar_constrained(per_K, M, n_memb, rho_limit):
    """기존 방식 (원논문 식12): argmax ECC s.t. rho_eff >= rho_Limit"""
    best = None
    for K, entries in per_K.items():
        res = eval_K(entries, M, n_memb)
        if res is None:
            continue
        ecc, rho, neff = res
        if rho < rho_limit:
            continue
        if best is None or ecc > best[1]:
            best = (K, ecc, neff, rho)
    return best


def find_kstar_combined(per_K, M, n_memb):
    """신규 방식: argmax [ECC(K) x rho_eff(K)], 제약 없음"""
    best = None
    for K, entries in per_K.items():
        res = eval_K(entries, M, n_memb)
        if res is None:
            continue
        ecc, rho, neff = res
        score = ecc * rho
        if best is None or score > best[1]:
            best = (K, score, ecc, rho, neff)
    return best   # (K*, score, ECC, rho_eff, N_eff)


# ============================================= 지속성 확인용
def get_top_clusters_obs(Z, boxes, K, n_memb, topN=3):
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
    D = make_desirability(y)(y)
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

    print('=' * 100)
    print(f'  기존(제약식, rho_Limit={RHO_LIMIT_BASELINE})  vs  신규(ECC x rho_eff 통합) 비교')
    print('=' * 100)
    print(f'  {"N_Memb":>7} | {"[기존] K*":>10} {"ECC":>8} {"rho_eff":>8} | '
          f'{"[신규] K*":>10} {"ECC":>8} {"rho_eff":>8} {"score":>8}')

    rows = []
    for n_memb in N_MEMB_OPTIONS:
        old = find_kstar_constrained(per_K, M, n_memb, RHO_LIMIT_BASELINE)
        new = find_kstar_combined(per_K, M, n_memb)
        if old is None or new is None:
            continue
        oK, oecc, oneff, orho = old
        nK, nscore, necc, nrho, nneff = new
        rows.append((n_memb, oK, oecc, orho, nK, necc, nrho, nscore))
        print(f'  {n_memb:>7} | {oK:>10} {oecc:>8.4f} {orho:>8.3f} | '
              f'{nK:>10} {necc:>8.4f} {nrho:>8.3f} {nscore:>8.4f}')

    # 전체 N_Memb 중 신규 방식에서 score가 최대인 조합을 "최종 채택"으로
    best_new = max(rows, key=lambda r: r[7])
    n_memb_star, oK, oecc, orho, nK, necc, nrho, nscore = best_new
    print(f'\n  ▶ 신규 방식 최종 채택: N_Memb*={n_memb_star}, K*={nK} '
          f'(ECC={necc:.4f}, rho_eff={nrho:.3f}, score={nscore:.4f})')
    print(f'  ▶ 참고: 같은 N_Memb에서 기존 방식은 K*={oK} '
          f'(ECC={oecc:.4f}, rho_eff={orho:.3f})')

    # ---------- 지속성 확인: 신규 K* 의 핵심 클러스터가 기존 K* 와 얼마나 겹치는지 ----------
    print('\n' + '=' * 100)
    print('  핵심 클러스터 지속성: 신규(통합목적함수) K* vs 기존(제약식) K*')
    print('=' * 100)
    old_top = get_top_clusters_obs(Z, boxes, oK, n_memb_star, topN=3)
    new_top = get_top_clusters_obs(Z, boxes, nK, n_memb_star, topN=3)
    print(f'  기존 K*={oK} 상위 클러스터 크기: {[len(c[1]) for c in old_top]}')
    print(f'  신규 K*={nK} 상위 클러스터 크기: {[len(c[1]) for c in new_top]}')
    print(f'\n  {"rank":>6} {"Jaccard(기존 vs 신규)":>22}')
    for i, (_, old_obs) in enumerate(old_top):
        best_j = max((jaccard_sets(old_obs, obs) for _, obs in new_top), default=0.0)
        print(f'  {i+1:>6} {best_j:>22.3f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
