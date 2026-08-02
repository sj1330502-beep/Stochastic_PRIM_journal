"""
================================================================================
아이디어 6번 — 방식4(c=30) vs 원논문, 합성데이터 5회(랜덤시드) 비교
================================================================================
RSM(실측 1030개로 학습, R^2=0.811)을 고정된 '알려진 desirability 관계식'
으로 삼고, 이 관계식에서 독립적인 합성 데이터셋을 5회 생성한다. 매 회마다
완전히 새로운(재현되지 않는) 무작위 시드를 사용하며, 그 시드값을 함께
출력해 필요시 재현할 수 있게 한다.

각 합성 데이터셋마다:
  - training/confirmation 분할
  - MRS-PRIM 박스 생성 + N_Memb(5~10) x K 완전탐색
  - [기존(rho>=0.6)] vs [방식4(desirability 포함, c=30)] 의 K* 를 각각 선정
  - 그 K*의 최대 클러스터를 confirmation으로 채점 (D_conf)
을 수행하고, 시드별 결과를 표로, 마지막에 종합(평균/승률)을 출력한다.
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
S_OPTIONS, T_PER_SIZE = (4, 5, 6), 667
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
METHOD4_ABC = dict(a=0.2, b=0.2, c=30.0)

TRAIN_RATIO = 0.7
N_SYNTH = 1030
N_REPEATS = 5


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
    boxes = []
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
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


def run_one_repeat(model, y_min, y_max, real_X, seed):
    X_synth = sample_synthetic_X(real_X, N_SYNTH, seed)
    D_synth = desirability_from_strength(rsm_predict(model, X_synth), y_min, y_max)

    rng_split = np.random.default_rng(seed + 999983)
    n = len(D_synth)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, D_train = X_synth[train_idx], D_synth[train_idx]
    X_confirm, D_confirm = X_synth[confirm_idx], D_synth[confirm_idx]

    boxes = build_boxes(X_train, D_train, seed)
    if len(boxes) < 10:
        return None
    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)

    res_orig = full_search_original(raw, M, k_grid, N_MEMB_OPTIONS)
    res_m4 = full_search_method4(raw, M, k_grid, N_MEMB_OPTIONS, **METHOD4_ABC)

    result = {}
    if res_orig is not None:
        n_memb_o, K_o, v_o = res_orig
        d_o, n_o = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, v_o['largest_mem'])
        result['기존(rho>=0.6)'] = dict(n_memb=n_memb_o, K=K_o, d_conf=d_o, n_conf=n_o)
    else:
        result['기존(rho>=0.6)'] = None

    if res_m4 is not None:
        n_memb_4, K_4, v_4 = res_m4
        d_4, n_4 = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, v_4['largest_mem'])
        result['방식4(c=30)'] = dict(n_memb=n_memb_4, K=K_4, d_conf=d_4, n_conf=n_4)
    else:
        result['방식4(c=30)'] = None

    return result


def main():
    real_X, real_y = load_real_Xy(REAL_CSV_PATH)
    print(f'실측 데이터 {len(real_y)}개로 RSM 적합 (관계식은 이후 고정) ...')
    model = fit_response_surface(real_X, real_y)
    y_min, y_max = real_y.min(), real_y.max()
    print(f'  R^2 = {model["r2"]:.3f}\n')

    print('=' * 90)
    print(f'  합성 데이터 {N_REPEATS}회 생성(매회 새로운 무작위 시드) -> 방식4 vs 원논문 비교')
    print('=' * 90)

    all_results = []
    for rep in range(N_REPEATS):
        seed = int(np.random.default_rng().integers(0, 1_000_000))
        print(f'\n[반복 {rep+1}/{N_REPEATS}] 시드={seed} 로 합성 데이터 생성 및 파이프라인 실행 ...')
        res = run_one_repeat(model, y_min, y_max, real_X, seed=seed)
        if res is None:
            print('  박스 생성 실패, 건너뜀')
            continue
        all_results.append((seed, res))
        for name, v in res.items():
            if v is None:
                print(f'  {name}: 조건 만족 조합 없음')
            else:
                print(f'  {name}: N_Memb*={v["n_memb"]} K*={v["K"]} '
                      f'D_conf={v["d_conf"]:.4f} n_conf={v["n_conf"]}')

    print('\n' + '=' * 90)
    print('  시드별 D_conf 요약표')
    print('=' * 90)
    print(f'  {"시드":>10} {"기존 D_conf":>12} {"방식4 D_conf":>13} {"차이(방식4-기존)":>16}')
    diffs = []
    for seed, res in all_results:
        o = res.get('기존(rho>=0.6)')
        m4 = res.get('방식4(c=30)')
        if o is None or m4 is None:
            continue
        diff = m4['d_conf'] - o['d_conf']
        diffs.append(diff)
        print(f'  {seed:>10} {o["d_conf"]:>12.4f} {m4["d_conf"]:>13.4f} {diff:>+16.4f}')

    print('\n' + '=' * 90)
    print('  종합 결론')
    print('=' * 90)
    if diffs:
        wins = sum(1 for d in diffs if d >= 0)
        print(f'  방식4가 기존과 같거나 능가한 횟수: {wins}/{len(diffs)}')
        print(f'  평균 차이(방식4 - 기존): {np.mean(diffs):+.4f}')
    else:
        print('  유효한 비교 결과가 없음')

    print('\n완료.')


if __name__ == '__main__':
    main()
