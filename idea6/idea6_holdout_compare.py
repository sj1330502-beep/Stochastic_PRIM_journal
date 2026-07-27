"""
================================================================================
아이디어 6번 — Hold-out 검증: 기존(제약식) vs 신규(통합목적함수) K* 비교
================================================================================
목적: "그래프 모양이 예쁘다"를 넘어, 각 방식이 고른 K*의 최대 클러스터가
      confirmation(한 번도 안 본 데이터)에서 실제로 더 나은 desirability를
      찾아주는지 직접 비교한다.

절차
  [0] 데이터를 training(70%)/confirmation(30%)으로 분할
  [1] training 데이터만으로 MRS-PRIM 박스 생성 및 클러스터링
  [2] 4가지 K* 선정 방식 각각 적용:
        - 기존: argmax ECC  s.t. rho_eff >= 0.6
        - 방식0: argmax ECC^0.5 * rho_eff^0.5
        - 방식1: argmax 가중조화평균(beta=1.0)
        - 방식3: argmax ECC - lambda*(K/M), lambda=1.0
  [3] 각 K*의 최대(멤버 최다) 효과적 클러스터 -> 그 멤버들의 관측치 union으로
      만든 통합 박스(변수별 min-max)를 confirmation 데이터에 적용해, 그
      범위 안에 든 confirmation 관측치의 실측 desirability 평균을 비교.
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
N_MEMB = 5
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60


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


# ============================================= MRS-PRIM 박스 생성 (training 만)
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


# ============================================= 클러스터링
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


# ============================================= 4가지 K* 선정 방식
def select_original(all_vals, rho_limit=RHO_LIMIT_BASELINE):
    best = None
    for K, ecc, rho in all_vals:
        if rho < rho_limit:
            continue
        if best is None or ecc > best[1]:
            best = (K, ecc, rho)
    return best


def select_method0(all_vals, w=0.5):
    best = None
    for K, ecc, rho in all_vals:
        score = ecc ** w * rho ** (1 - w)
        if best is None or score > best[1]:
            best = (K, ecc, rho)
    return best


def select_method1(all_vals, beta=1.0):
    best = None
    for K, ecc, rho in all_vals:
        denom = beta ** 2 * ecc + rho
        score = (1 + beta ** 2) * ecc * rho / denom if denom > 0 else 0
        if best is None or score > best[1]:
            best = (K, ecc, rho)
    return best


def select_method3(all_vals, M, lam=1.0):
    best = None
    for K, ecc, rho in all_vals:
        score = ecc - lam * (K / M)
        if best is None or score > best[1]:
            best = (K, ecc, rho)
    return best


# ============================================= 최대 클러스터의 통합박스 + Hold-out
def get_largest_cluster_box(Z, boxes, X_train, K, n_memb):
    lab = fcluster(Z, K, criterion='maxclust')
    clusters = []
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        if len(mem) < n_memb:
            continue
        clusters.append(mem)
    clusters.sort(key=lambda m: -len(m))
    largest = clusters[0]
    all_idx = np.concatenate([boxes[i]['idx'] for i in largest])
    lo, hi = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
    return lo, hi, len(largest)


def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


# ============================================= 실행
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

    print('클러스터링(경험적 Jaccard -> average-linkage) ...')
    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    per_K = precompute_per_K(Z, sets, M, k_grid)

    all_vals = []
    for K in k_grid:
        res = eval_K(per_K[K], M, N_MEMB)
        if res is not None:
            all_vals.append((K, res[0], res[1]))
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}\n')

    print('[2~3] 4가지 K* 선정 방식 -> 최대 클러스터 -> Hold-out 검증')
    methods = {
        '기존(rho>=0.6)': select_original(all_vals),
        '방식0(단순곱셈)': select_method0(all_vals),
        '방식1(조화평균)': select_method1(all_vals),
        '방식3(AIC/BIC페널티)': select_method3(all_vals, M),
    }

    print(f'\n  {"방식":>18} {"K*":>6} {"ECC":>8} {"rho":>8} {"n_mem":>6} '
          f'{"n_conf":>7} {"D_conf(실측)":>12}')
    results = []
    for name, (K, ecc, rho) in methods.items():
        lo, hi, n_mem = get_largest_cluster_box(Z, boxes, X_train, K, N_MEMB)
        d_conf, n_conf = holdout_eval(lo, hi, X_confirm, D_confirm)
        results.append((name, K, ecc, rho, n_mem, n_conf, d_conf))
        print(f'  {name:>18} {K:>6} {ecc:>8.4f} {rho:>8.3f} {n_mem:>6} '
              f'{n_conf:>7} {d_conf:>12.4f}')

    print('\n' + '=' * 70)
    print('  결론')
    print('=' * 70)
    baseline_d = results[0][6]
    best_new = max(results[1:], key=lambda r: r[6])
    print(f'  기존 방식 confirmation D = {baseline_d:.4f}')
    print(f'  신규 방식 중 최고: {best_new[0]}, confirmation D = {best_new[6]:.4f} '
          f'(차이 {best_new[6]-baseline_d:+.4f})')
    if best_new[6] > baseline_d:
        print('  -> 신규 방식이 confirmation 기준으로도 더 나은 조건을 찾음')
    else:
        print('  -> 신규 방식이 그래프 형태는 개선했으나, confirmation 성능은')
        print('     기존과 비슷하거나 못함 (추가 검토 필요)')

    print('\n완료.')


if __name__ == '__main__':
    main()
