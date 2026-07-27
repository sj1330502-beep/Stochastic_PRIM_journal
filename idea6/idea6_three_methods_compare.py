"""
================================================================================
아이디어 6번 확장 — ECC/rho_eff 통합 목적함수 3가지 방식 비교
================================================================================
방식 0. 단순 곱셈       : score = ECC^w * rho^(1-w)
방식 1. 가중조화평균    : score = (1+beta^2)*ECC*rho / (beta^2*ECC + rho)
방식 2. 정규화 후 곱셈  : (ECC를 K격자 내 최소~최대로 0~1 정규화) x
                          (rho를 K격자 내 최소~최대로 0~1 정규화)

각 방식·파라미터에서 나온 K*가, 기준(단순곱셈 w=0.5, K*=310)의 상위 1~3위
클러스터와 관측치 집합 기준 얼마나 겹치는지(Jaccard)로 안정성을 비교한다.
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
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 3
N_MEMB = 5                 # 이전 실험들에서 일관되게 최적이었던 값으로 고정
K_GRID_STEP = 5

W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]         # 방식 0(단순곱셈)의 ECC 가중치
BETA_GRID = [0.5, 0.7, 1.0, 1.4, 2.0]      # 방식 1(조화평균)의 beta


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
    eff = [(n, g) for n, g in entries if n >= n_memb]
    if not eff:
        return None
    rho = sum(n for n, _ in eff) / M
    ecc = np.mean([g for _, g in eff])
    return ecc, rho


# ============================================= 세 가지 통합 목적함수
def method0_product(all_vals, w):
    """단순 곱셈: ECC^w * rho^(1-w)"""
    best = None
    for K, ecc, rho in all_vals:
        score = ecc ** w * rho ** (1 - w)
        if best is None or score > best[1]:
            best = (K, score, ecc, rho)
    return best


def method1_harmonic(all_vals, beta):
    """가중조화평균 (F-beta 스타일)"""
    best = None
    for K, ecc, rho in all_vals:
        denom = (beta ** 2) * ecc + rho
        score = (1 + beta ** 2) * ecc * rho / denom if denom > 0 else 0.0
        if best is None or score > best[1]:
            best = (K, score, ecc, rho)
    return best


def method2_normalized(all_vals):
    """정규화 후 곱셈: K격자 내 min-max로 각각 0~1 정규화 후 곱함"""
    eccs = np.array([v[1] for v in all_vals])
    rhos = np.array([v[2] for v in all_vals])
    e_lo, e_hi = eccs.min(), eccs.max()
    r_lo, r_hi = rhos.min(), rhos.max()
    best = None
    for K, ecc, rho in all_vals:
        ecc_n = (ecc - e_lo) / (e_hi - e_lo) if e_hi > e_lo else 1.0
        rho_n = (rho - r_lo) / (r_hi - r_lo) if r_hi > r_lo else 1.0
        score = ecc_n * rho_n
        if best is None or score > best[1]:
            best = (K, score, ecc, rho)
    return best, (e_lo, e_hi, r_lo, r_hi)


# ============================================= 지속성(안정성) 확인
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


def persistence(Z, boxes, K, n_memb, baseline_top):
    top = get_top_clusters_obs(Z, boxes, K, n_memb, topN=3)
    return [max((jaccard_sets(b, o) for _, o in top), default=0.0) for _, b in baseline_top]


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
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    per_K = precompute_per_K(Z, sets, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]} (step {K_GRID_STEP})\n')

    all_vals = []
    for K in k_grid:
        res = eval_K(per_K[K], M, N_MEMB)
        if res is not None:
            all_vals.append((K, res[0], res[1]))   # (K, ECC, rho)

    # ---------- 기준선: 단순곱셈 w=0.5 ----------
    baseline = method0_product(all_vals, 0.5)
    base_K = baseline[0]
    baseline_top = get_top_clusters_obs(Z, boxes, base_K, N_MEMB, topN=3)
    print(f'기준(방식0, w=0.5): K*={base_K}, ECC={baseline[2]:.4f}, rho={baseline[3]:.3f}')
    print(f'기준 상위3 클러스터 크기: {[len(c[1]) for c in baseline_top]}\n')

    print('=' * 90)
    print('  방식 0: 단순 곱셈  score = ECC^w * rho^(1-w)')
    print('=' * 90)
    print(f'  {"w":>6} {"K*":>6} {"ECC":>8} {"rho":>8} {"score":>8} | '
          f'{"rank1":>7} {"rank2":>7} {"rank3":>7}')
    for w in W_GRID:
        K, score, ecc, rho = method0_product(all_vals, w)
        jv = persistence(Z, boxes, K, N_MEMB, baseline_top)
        print(f'  {w:>6.1f} {K:>6} {ecc:>8.4f} {rho:>8.3f} {score:>8.4f} | '
              f'{jv[0]:>7.3f} {jv[1]:>7.3f} {jv[2]:>7.3f}')

    print('\n' + '=' * 90)
    print('  방식 1: 가중조화평균  score = (1+beta^2)*ECC*rho / (beta^2*ECC+rho)')
    print('=' * 90)
    print(f'  {"beta":>6} {"K*":>6} {"ECC":>8} {"rho":>8} {"score":>8} | '
          f'{"rank1":>7} {"rank2":>7} {"rank3":>7}')
    for beta in BETA_GRID:
        K, score, ecc, rho = method1_harmonic(all_vals, beta)
        jv = persistence(Z, boxes, K, N_MEMB, baseline_top)
        print(f'  {beta:>6.1f} {K:>6} {ecc:>8.4f} {rho:>8.3f} {score:>8.4f} | '
              f'{jv[0]:>7.3f} {jv[1]:>7.3f} {jv[2]:>7.3f}')

    print('\n' + '=' * 90)
    print('  방식 2: 정규화 후 곱셈  score = ECC_norm * rho_norm')
    print('=' * 90)
    (K, score, ecc, rho), (e_lo, e_hi, r_lo, r_hi) = method2_normalized(all_vals)
    jv = persistence(Z, boxes, K, N_MEMB, baseline_top)
    print(f'  ECC 범위=[{e_lo:.3f},{e_hi:.3f}]  rho 범위=[{r_lo:.3f},{r_hi:.3f}]')
    print(f'  K*={K}  ECC={ecc:.4f}  rho={rho:.3f}  score={score:.4f} | '
          f'rank1={jv[0]:.3f}  rank2={jv[1]:.3f}  rank3={jv[2]:.3f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
