"""
================================================================================
아이디어 6번 — N_Memb x K 완전탐색 (콘크리트 데이터)
================================================================================
이전 실험(idea6_three_methods_compare.py 이후)들은 N_Memb=5로 고정하고
진행했으나, 원논문 식(12)는 K*가 N_Memb 의 함수이기도 하다. 이번 코드는
기존/방식0/방식3/방식4 각각에 대해 N_Memb(5~10) x K 전체를 탐색해, 그
방식 고유의 목적함수를 최대화하는 (N_Memb*, K*) 조합을 자동으로 찾는다.

효율화: fcluster(Z, K)는 K별로 1회만 계산해 원본 클러스터(멤버 인덱스 목록)를
저장해두고, N_Memb 별 ECC/rho/Dbar 계산은 그 저장된 목록을 멤버수로 필터링만
해서 구한다 (재계산 없음).
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

METHOD4_ABC = dict(a=0.2, b=0.2, c=30.0)


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
            gamma = len(I) / len(U) if U else 0.0
            dbar = np.mean([boxes[i]['dbar'] for i in mem])
            clusters.append(dict(mem=mem, n=len(mem), gamma=gamma, dbar=dbar))
        raw[K] = clusters
    return raw


def eval_for_nmemb(raw_clusters_at_K, n_memb, M):
    eff = [c for c in raw_clusters_at_K if c['n'] >= n_memb]
    if not eff:
        return None
    rho = sum(c['n'] for c in eff) / M
    ecc = np.mean([c['gamma'] for c in eff])
    largest = max(eff, key=lambda c: c['n'])
    return dict(ecc=ecc, rho=rho, dbar=largest['dbar'], largest_mem=largest['mem'],
               n_mem=largest['n'])


def full_search_original(raw, M, k_grid, n_memb_options, rho_limit=RHO_LIMIT_BASELINE):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None or v['rho'] < rho_limit:
                continue
            if best is None or v['ecc'] > best[2]['ecc']:
                best = (n_memb, K, v)
    return best


def full_search_method0(raw, M, k_grid, n_memb_options, w=0.5):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None:
                continue
            score = v['ecc'] ** w * v['rho'] ** (1 - w)
            if best is None or score > best[3]:
                best = (n_memb, K, v, score)
    return best[:3] if best else None


def full_search_method1(raw, M, k_grid, n_memb_options, beta=1.0):
    """가중조화평균(F-beta 스타일). beta=1.0이면 방식0(w=0.5)과 자주 수렴하는지 확인 대상."""
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None:
                continue
            denom = beta ** 2 * v['ecc'] + v['rho']
            score = (1 + beta ** 2) * v['ecc'] * v['rho'] / denom if denom > 0 else 0.0
            if best is None or score > best[3]:
                best = (n_memb, K, v, score)
    return best[:3] if best else None


def full_search_method3(raw, M, k_grid, n_memb_options, lam=1.0):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None:
                continue
            score = v['ecc'] - lam * (K / M)
            if best is None or score > best[3]:
                best = (n_memb, K, v, score)
    return best[:3] if best else None


def full_search_method4(raw, M, k_grid, n_memb_options, a, b, c):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None:
                continue
            d = max(v['dbar'], 1e-6)
            score = (v['ecc'] ** a) * (v['rho'] ** b) * (d ** c)
            if best is None or score > best[3]:
                best = (n_memb, K, v, score)
    return best[:3] if best else None


def holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, largest_mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in largest_mem])
    lo, hi = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


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

    print('클러스터링 및 K별 원본 클러스터 계산 (N_Memb 필터링은 재사용) ...')
    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}, N_Memb 후보 {N_MEMB_OPTIONS}\n')

    print('=' * 90)
    print('  4가지 방식의 (N_Memb x K) 완전탐색 -> Hold-out 비교')
    print('=' * 90)

    results_raw = {
        '기존(rho>=0.6)': full_search_original(raw, M, k_grid, N_MEMB_OPTIONS),
        '방식0(단순곱셈)': full_search_method0(raw, M, k_grid, N_MEMB_OPTIONS),
        '방식1(조화평균)': full_search_method1(raw, M, k_grid, N_MEMB_OPTIONS),
        '방식3(AIC/BIC)': full_search_method3(raw, M, k_grid, N_MEMB_OPTIONS),
        '방식4(desirability포함)': full_search_method4(
            raw, M, k_grid, N_MEMB_OPTIONS, **METHOD4_ABC),
    }

    print(f'\n  {"방식":>22} {"N_Memb*":>8} {"K*":>6} {"ECC":>8} {"rho":>8} '
          f'{"n_conf":>7} {"D_conf(실측)":>12}')
    results = []
    for name, res in results_raw.items():
        if res is None:
            print(f'  {name:>22}   (조건 만족 조합 없음)')
            continue
        n_memb, K, v = res
        d_conf, n_conf = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm,
                                              v['largest_mem'])
        results.append((name, n_memb, K, v['ecc'], v['rho'], n_conf, d_conf))
        print(f'  {name:>22} {n_memb:>8} {K:>6} {v["ecc"]:>8.4f} {v["rho"]:>8.3f} '
              f'{n_conf:>7} {d_conf:>12.4f}')

    print('\n' + '=' * 90)
    print('  결론')
    print('=' * 90)
    if results:
        baseline = [r for r in results if r[0] == '기존(rho>=0.6)']
        if baseline:
            base_d = baseline[0][6]
            others = [r for r in results if r[0] != '기존(rho>=0.6)']
            if others:
                best_new = max(others, key=lambda r: r[6])
                print(f'  기존 방식 confirmation D = {base_d:.4f} (N_Memb*={baseline[0][1]})')
                print(f'  신규 방식 중 최고: {best_new[0]}, D = {best_new[6]:.4f} '
                      f'(N_Memb*={best_new[1]}, 차이 {best_new[6]-base_d:+.4f})')

    print('\n완료.')


if __name__ == '__main__':
    main()
