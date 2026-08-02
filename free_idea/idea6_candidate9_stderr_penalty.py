"""
================================================================================
아이디어 6번 후보 9 — 표준오차(신뢰구간 하한) 벌점 목적함수
================================================================================
후보5(부피 벌점)가 실패한 이유: Dbar와 부피가 K에 대해 같은 방향(K가 클수록
둘 다 '좋아짐')으로 움직여서, 벌점이 desirability 상승과 상쇄되지 못했다.

후보9는 다른 종류의 벌점을 쓴다: '그 클러스터의 desirability 평균이 통계적
으로 얼마나 신뢰할 만한가(표준오차)'.
    score(K) = Dbar_largest(K) - z * (s_D / sqrt(n))
  s_D : 최대 클러스터에 속한 training 관측치들의 desirability 표준편차
  n   : 그 관측치 수(고유 개수, box 개수 아님)
  z   : 벌점 강도 (신뢰구간 계수처럼 해석 가능)

desirability 평균 자체는 K가 클수록 계속 오르지만, 표본이 지나치게 적어지면
표준오차가 급격히 커진다 -- 이는 '부피/개수 감소'와는 독립적인, desirability
상승과 상쇄될 수 있는 진짜 트레이드오프일 가능성이 있다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

TRAIN_RATIO = 0.7
SPLIT_SEED = 0

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]

Z_GRID = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


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


def precompute_raw_clusters(Z, sets, boxes, M, k_grid):
    raw = {}
    for K in k_grid:
        lab = fcluster(Z, K, criterion='maxclust')
        clusters = []
        for k in np.unique(lab):
            mem = np.where(lab == k)[0]
            I = set.intersection(*[sets[m] for m in mem])
            U = set.union(*[sets[m] for m in mem])
            gamma_val = len(I) / len(U) if U else 0.0
            dbar = np.mean([boxes[i]['dbar'] for i in mem])
            clusters.append(dict(mem=mem, n=len(mem), gamma=gamma_val, dbar=dbar,
                                 union_idx=U))
        raw[K] = clusters
    return raw


def effective_clusters(raw_clusters_at_K, n_memb):
    return [c for c in raw_clusters_at_K if c['n'] >= n_memb]


def largest_of(eff):
    return max(eff, key=lambda c: c['n'])


def cluster_range(boxes, X_train, mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    return X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)


def stderr_of_cluster(union_idx, D_train):
    idx_arr = np.array(sorted(union_idx))
    vals = D_train[idx_arr]
    n = len(vals)
    if n < 2:
        return 0.0, n
    s = np.std(vals, ddof=1)
    return s / np.sqrt(n), n


def eval_ecc_rho(eff, M):
    if not eff:
        return None
    rho = sum(c['n'] for c in eff) / M
    ecc = np.mean([c['gamma'] for c in eff])
    return dict(ecc=ecc, rho=rho)


def eval_range_on_confirm(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


def full_search_original(raw, M, k_grid, n_memb_options, rho_limit=RHO_LIMIT_BASELINE):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            eff = effective_clusters(raw[K], n_memb)
            v = eval_ecc_rho(eff, M)
            if v is None or v['rho'] < rho_limit:
                continue
            if best is None or v['ecc'] > best[2]['ecc']:
                best = (n_memb, K, v, largest_of(eff)['mem'])
    return best


def full_search_candidate9(raw, D_train, k_grid, n_memb_options, z):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            eff = effective_clusters(raw[K], n_memb)
            if not eff:
                continue
            largest = largest_of(eff)
            se, n_obs = stderr_of_cluster(largest['union_idx'], D_train)
            score = largest['dbar'] - z * se
            if best is None or score > best[4]:
                best = (n_memb, K, largest['mem'], (se, n_obs), score)
    return best


def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X_all = df.iloc[:, :8].values.astype(float)
    y_all = df.iloc[:, 8].values.astype(float)

    rng_split = np.random.default_rng(SPLIT_SEED)
    n = len(y_all)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_confirm = X_all[confirm_idx]

    d_func = make_desirability(y_all)
    D_train = d_func(y_train)
    D_confirm = d_func(y_all[confirm_idx])
    print(f'[0] 분할: training {len(y_train)}개 / confirmation {len(X_confirm)}개\n')

    print('[1] training 데이터로 MRS-PRIM 박스 생성')
    boxes = build_boxes(X_train, D_train)
    print(f'  박스 {len(boxes)}개\n')

    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}\n')

    res_orig = full_search_original(raw, M, k_grid, N_MEMB_OPTIONS)
    n_memb_o, K_o, v_o, mem_o = res_orig
    lo_o, hi_o = cluster_range(boxes, X_train, mem_o)
    d_o, n_o = eval_range_on_confirm(lo_o, hi_o, X_confirm, D_confirm)
    print('=' * 90)
    print(f'  기존(rho>=0.6): N_Memb*={n_memb_o}, K*={K_o}, D_conf={d_o:.4f} (n={n_o})')
    print('=' * 90)

    print(f'\n  {"z":>7} {"N_Memb*":>8} {"K*":>6} {"SE":>8} {"n_obs":>7} '
          f'{"n_conf":>7} {"D_conf":>8} {"기존대비":>9}')
    results = []
    for z in Z_GRID:
        res = full_search_candidate9(raw, D_train, k_grid, N_MEMB_OPTIONS, z)
        n_memb, K, mem, (se, n_obs), score = res
        lo, hi = cluster_range(boxes, X_train, mem)
        d_conf, n_conf = eval_range_on_confirm(lo, hi, X_confirm, D_confirm)
        diff = d_conf - d_o
        results.append((z, n_memb, K, se, n_obs, n_conf, d_conf, diff))
        print(f'  {z:>7.3f} {n_memb:>8} {K:>6} {se:>8.4f} {n_obs:>7} '
              f'{n_conf:>7} {d_conf:>8.4f} {diff:>+9.4f}')

    print('\n' + '=' * 90)
    print('  결론')
    print('=' * 90)
    best = max(results, key=lambda r: r[6])
    print(f'  기존 방식        : D_conf = {d_o:.4f}')
    print(f'  후보9 최고(z={best[0]}) : D_conf = {best[6]:.4f} (차이 {best[7]:+.4f})')
    if best[6] > d_o:
        print('  -> 후보9가 기존을 능가함')
    else:
        print('  -> 후보9도 기존을 넘지 못함')

    print('\n완료.')


if __name__ == '__main__':
    main()
