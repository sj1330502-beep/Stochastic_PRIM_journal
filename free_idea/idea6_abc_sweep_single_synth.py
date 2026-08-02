"""
================================================================================
아이디어 6번 — 합성데이터 1개로 방식4의 a,b,c 스윕 후 원논문과 비교
================================================================================
합성 데이터셋을 1개만 생성(시간 절약)하고, 그 안에서 방식4의 (a,b,c) 조합을
그리드로 스윕해 confirmation D_conf 기준 최고 조합을 찾아 원논문(기존
rho>=0.6)과 비교한다.

주의: 하나의 confirmation 세트로 여러 (a,b,c) 조합을 채점하고 그중 최고를
고르는 방식이라, 선택된 조합의 성능에는 다중비교로 인한 낙관 편향이
섞일 수 있다. 이는 결과 해석 시 감안해야 한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

FEAT_NAMES = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
              'CoarseAgg', 'FineAgg', 'Age']
AGE_COL = FEAT_NAMES.index('Age')
ZERO_INFLATION_THRESHOLD = 0.05
JITTER_STD_RATIO = 0.03
TARGET_STRENGTH = 60.0

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE = (4, 5, 6), 300
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]

A_GRID = [0.1, 0.2, 0.3]
B_GRID = [0.1, 0.2, 0.3]
C_GRID = [10, 20, 30, 50, 70]

TRAIN_RATIO = 0.7
N_SYNTH = 1030


def load_real_Xy(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df.iloc[:, :8].values.astype(float), df.iloc[:, 8].values.astype(float)


def fit_response_surface(X, y):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / sigma
    N, P = Xs.shape
    terms = [np.ones(N)] + [Xs[:, i] for i in range(P)] + [Xs[:, i] ** 2 for i in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            terms.append(Xs[:, i] * Xs[:, j])
    Design = np.column_stack(terms)
    coef, *_ = np.linalg.lstsq(Design, y, rcond=None)
    y_hat = Design @ coef
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(coef=coef, mu=mu, sigma=sigma, P=P, r2=r2)


def rsm_predict(model, X):
    mu, sigma, coef, P = model['mu'], model['sigma'], model['coef'], model['P']
    Xs = (np.atleast_2d(X) - mu) / sigma
    N = Xs.shape[0]
    terms = [np.ones(N)] + [Xs[:, i] for i in range(P)] + [Xs[:, i] ** 2 for i in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            terms.append(Xs[:, i] * Xs[:, j])
    return np.column_stack(terms) @ coef


def desirability_from_strength(y_pred, y_min, y_max, target=TARGET_STRENGTH):
    y_pred = np.atleast_1d(y_pred)
    d = np.zeros_like(y_pred)
    m1 = (y_pred >= y_min) & (y_pred <= target)
    d[m1] = (y_pred[m1] - y_min) / (target - y_min)
    m2 = (y_pred > target) & (y_pred <= y_max)
    d[m2] = (y_max - y_pred[m2]) / (y_max - target)
    return np.clip(d, 0.0, 1.0)


def zero_inflated_columns(real_X, threshold=ZERO_INFLATION_THRESHOLD):
    return [p for p in range(real_X.shape[1])
            if p != AGE_COL and np.mean(real_X[:, p] == 0) >= threshold]


def jitter_continuous(row, real_X, zero_cols, rng, jitter_std_ratio=JITTER_STD_RATIO):
    new_row = row.copy()
    stds = real_X.std(axis=0)
    for p in range(len(row)):
        if p == AGE_COL:
            continue
        if p in zero_cols and row[p] == 0:
            continue
        new_row[p] = np.clip(row[p] + rng.normal(0, jitter_std_ratio * stds[p]),
                             real_X[:, p].min(), real_X[:, p].max())
    return new_row


def build_recipe_pool(real_X):
    non_age_idx = [p for p in range(real_X.shape[1]) if p != AGE_COL]
    keys = [tuple(np.round(real_X[i, non_age_idx], 6)) for i in range(len(real_X))]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    recipes = [np.array(real_X[idxs[0]]) for idxs in groups.values()]
    return recipes, list(groups.values())


def sample_synthetic_X(real_X, N, seed):
    rng = np.random.default_rng(seed)
    zero_cols = zero_inflated_columns(real_X)
    recipes, recipe_row_idx = build_recipe_pool(real_X)
    n_recipes = len(recipes)
    replicate_counts = np.array([len(idxs) for idxs in recipe_row_idx])
    ages_g, cnt_g = np.unique(real_X[:, AGE_COL], return_counts=True)
    probs_g = cnt_g / cnt_g.sum()

    rows = []
    while len(rows) < N:
        r = rng.integers(0, n_recipes)
        recipe = recipes[r]
        n_rep = max(1, int(rng.choice(replicate_counts)))
        own_ages = real_X[recipe_row_idx[r], AGE_COL]
        jrecipe = jitter_continuous(recipe, real_X, zero_cols, rng)
        for _ in range(n_rep):
            if len(rows) >= N:
                break
            row = jrecipe.copy()
            row[AGE_COL] = rng.choice(own_ages) if len(own_ages) > 1 else rng.choice(ages_g, p=probs_g)
            rows.append(row)
    return np.array(rows[:N])


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


def build_boxes(X, D, seed):
    rng = np.random.default_rng(seed)
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 200 == 0:
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
    return dict(ecc=ecc, rho=rho, dbar=largest['dbar'], largest_mem=largest['mem'])


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
    real_X, real_y = load_real_Xy(REAL_CSV_PATH)
    print('실측 데이터로 RSM 적합 (관계식 고정) ...')
    model = fit_response_surface(real_X, real_y)
    y_min, y_max = real_y.min(), real_y.max()
    print(f'  R^2 = {model["r2"]:.3f}\n')

    seed = int(np.random.default_rng().integers(0, 1_000_000))
    print(f'[0] 합성 데이터 1개 생성 (시드={seed})')
    X_synth = sample_synthetic_X(real_X, N_SYNTH, seed)
    D_synth = desirability_from_strength(rsm_predict(model, X_synth), y_min, y_max)

    rng_split = np.random.default_rng(seed + 999983)
    n = len(D_synth)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, D_train = X_synth[train_idx], D_synth[train_idx]
    X_confirm, D_confirm = X_synth[confirm_idx], D_synth[confirm_idx]
    print(f'  training {len(X_train)}개 / confirmation {len(X_confirm)}개\n')

    print('[1] training 데이터로 MRS-PRIM 박스 생성')
    boxes = build_boxes(X_train, D_train, seed)
    print(f'  박스 {len(boxes)}개\n')

    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}\n')

    res_orig = full_search_original(raw, M, k_grid, N_MEMB_OPTIONS)
    n_memb_o, K_o, v_o = res_orig
    d_o, n_o = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, v_o['largest_mem'])
    print('=' * 90)
    print(f'  기존(rho>=0.6): N_Memb*={n_memb_o}, K*={K_o}, D_conf={d_o:.4f} (n={n_o})')
    print('=' * 90)

    print(f'\n[2] 방식4 (a,b,c) 그리드 스윕 ({len(A_GRID)}x{len(B_GRID)}x{len(C_GRID)}'
          f'={len(A_GRID)*len(B_GRID)*len(C_GRID)}개 조합)')
    grid_results = []
    for a in A_GRID:
        for b in B_GRID:
            for c in C_GRID:
                res = full_search_method4(raw, M, k_grid, N_MEMB_OPTIONS, a, b, c)
                if res is None:
                    continue
                n_memb, K, v = res
                d_conf, n_conf = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm,
                                                      v['largest_mem'])
                grid_results.append((a, b, c, n_memb, K, d_conf, n_conf))

    grid_results.sort(key=lambda r: -r[5])
    print(f'\n  상위 10개 조합 (a, b, c, N_Memb*, K*, D_conf, n_conf):')
    for r in grid_results[:10]:
        print(f'    a={r[0]}, b={r[1]}, c={r[2]:>3} | N_Memb*={r[3]} K*={r[4]:>4} '
              f'D_conf={r[5]:.4f} (n={r[6]})')

    best = grid_results[0]
    print('\n' + '=' * 90)
    print('  최종 비교')
    print('=' * 90)
    print(f'  기존(rho>=0.6)                    : D_conf = {d_o:.4f}')
    print(f'  방식4 최고(a={best[0]},b={best[1]},c={best[2]}) : D_conf = {best[5]:.4f} '
          f'(차이 {best[5]-d_o:+.4f})')
    if best[5] > d_o:
        print('  -> 방식4(스윕)가 기존을 능가함')
    elif best[5] == d_o:
        print('  -> 방식4(스윕)가 기존과 동률')
    else:
        print('  -> 방식4(스윕)도 기존을 넘지 못함')

    print('\n  [주의] 이 스윕은 하나의 confirmation 세트로 여러 조합을 채점한 것이라,')
    print('  "가장 잘 맞은 조합"이 우연일 가능성이 있다. 다른 시드/데이터에서 재검증 권장.')

    print('\n완료.')


if __name__ == '__main__':
    main()
