"""
================================================================================
합성 데이터 정답 기반 검증 — 6차 버전 (실측 기반 반응표면모델 desirability)
================================================================================
5차까지의 한계
  X(입력)는 실제 콘크리트의 구조(zero-inflation, Age 범주형, 레시피-반복측정)를
  충실히 재현했으나, D(desirability)는 여전히 우리가 임의로 심은 가우시안
  봉우리였다. 즉 "이 재료 배합이면 실제로 이런 강도가 나온다"는 물리적 관계가
  전혀 반영되지 않았고, 정답 위치도 완전 무작위로 정해졌다.

6차 개선
  desirability를 가우시안 봉우리 대신, 실제 1030개 관측치(재료배합->압축강도)로
  학습시킨 2차 반응표면모델(RSM: 주효과 + 이차항 + 교호작용)로 대체한다.
  이는 콘크리트공학에서 실제로 쓰이는 강도예측 모델링 방식(다중회귀/RSM)과
  같은 계열이다. 이 회귀식을 '실측에 기반한 알려진 진짜 관계식'으로 삼고,
  그 식을 수치최적화해서 desirability가 최대가 되는 지점(들)을 정답으로 규정한다.
  이렇게 하면:
    - D가 실측 데이터로 학습된 관계식이라는 점에서 '물리적으로 그럴듯함'을 확보하고
    - 그 관계식의 최적점을 수식으로 정확히 구할 수 있으므로 '정답을 안다'는
      조건(채점 가능성)도 유지된다.
  X 생성(레시피-반복측정 부트스트랩)은 5차 버전 그대로 재사용한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

FEAT_NAMES = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
              'CoarseAgg', 'FineAgg', 'Age']
AGE_COL = FEAT_NAMES.index('Age')
ZERO_INFLATION_THRESHOLD = 0.05
JITTER_STD_RATIO = 0.03

ALPHA_PEEL = 0.05
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20
S_RATIO_LOW, S_RATIO_HIGH = 0.5, 0.75
T_PER_SIZE = 300
MATCH_RADIUS_RATIO = 0.20

TARGET_STRENGTH = 60.0   # NTB 목표 (Koo et al. 콘크리트 케이스와 동일)


# ============================================= [0] 실측 데이터 로드 + 반응표면모델(RSM) 적합
def load_real_Xy(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    return X, y


def fit_response_surface(X, y):
    """
    2차 반응표면모델(RSM): y ~ 주효과(8) + 이차항(8) + 전체 교호작용(28)
    표준화 후 최소제곱으로 적합. 계수와 표준화 통계량을 함께 반환.
    """
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / sigma
    N, P = Xs.shape

    terms = [np.ones(N)]                                   # 절편
    terms += [Xs[:, i] for i in range(P)]                  # 주효과
    terms += [Xs[:, i] ** 2 for i in range(P)]             # 이차항
    for i in range(P):                                      # 교호작용
        for j in range(i + 1, P):
            terms.append(Xs[:, i] * Xs[:, j])
    Design = np.column_stack(terms)

    coef, *_ = np.linalg.lstsq(Design, y, rcond=None)
    y_hat = Design @ coef
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    return dict(coef=coef, mu=mu, sigma=sigma, P=P, r2=r2)


def rsm_predict(model, X):
    """적합된 RSM으로 X(N,P)의 강도를 예측."""
    mu, sigma, coef, P = model['mu'], model['sigma'], model['coef'], model['P']
    Xs = (np.atleast_2d(X) - mu) / sigma
    N = Xs.shape[0]
    terms = [np.ones(N)]
    terms += [Xs[:, i] for i in range(P)]
    terms += [Xs[:, i] ** 2 for i in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            terms.append(Xs[:, i] * Xs[:, j])
    Design = np.column_stack(terms)
    return Design @ coef


def desirability_from_strength(y_pred, y_real_min, y_real_max, target=TARGET_STRENGTH):
    """NTB(Nominal-The-Better) desirability, 원논문 Koo et al. Table1 방식과 동일."""
    y_pred = np.atleast_1d(y_pred)
    d = np.zeros_like(y_pred)
    m1 = (y_pred >= y_real_min) & (y_pred <= target)
    d[m1] = (y_pred[m1] - y_real_min) / (target - y_real_min)
    m2 = (y_pred > target) & (y_pred <= y_real_max)
    d[m2] = (y_real_max - y_pred[m2]) / (y_real_max - target)
    return np.clip(d, 0.0, 1.0)


# ============================================= [1] X 생성 (5차 버전 그대로 재사용)
def zero_inflated_columns(real_X, threshold=ZERO_INFLATION_THRESHOLD):
    P = real_X.shape[1]
    zero_cols = []
    for p in range(P):
        if p == AGE_COL:
            continue
        if np.mean(real_X[:, p] == 0) >= threshold:
            zero_cols.append(p)
    return zero_cols


def jitter_continuous(row, real_X, zero_cols, rng, jitter_std_ratio=JITTER_STD_RATIO):
    P = len(row)
    new_row = row.copy()
    stds = real_X.std(axis=0)
    for p in range(P):
        if p == AGE_COL:
            continue
        if p in zero_cols and row[p] == 0:
            continue   # 정확히 0인 값은 지터하지 않음 (zero-inflation 보존)
        new_row[p] = row[p] + rng.normal(0, jitter_std_ratio * stds[p])
        new_row[p] = np.clip(new_row[p], real_X[:, p].min(), real_X[:, p].max())
    return new_row


def build_recipe_pool(real_X):
    non_age_idx = [p for p in range(real_X.shape[1]) if p != AGE_COL]
    keys = [tuple(np.round(real_X[i, non_age_idx], 6)) for i in range(len(real_X))]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    recipes = [np.array(real_X[idxs[0]]) for idxs in groups.values()]
    recipe_row_idx = list(groups.values())
    return recipes, recipe_row_idx, non_age_idx


def sample_recipe_replicate_inputs(real_X, N, seed, jitter_std_ratio=JITTER_STD_RATIO):
    rng = np.random.default_rng(seed)
    zero_cols = zero_inflated_columns(real_X)
    recipes, recipe_row_idx, _ = build_recipe_pool(real_X)
    n_recipes = len(recipes)
    replicate_counts = np.array([len(idxs) for idxs in recipe_row_idx])

    ages_global, age_counts_global = np.unique(real_X[:, AGE_COL], return_counts=True)
    age_probs_global = age_counts_global / age_counts_global.sum()

    rows = []
    while len(rows) < N:
        r = rng.integers(0, n_recipes)
        recipe = recipes[r]
        n_rep = max(1, int(rng.choice(replicate_counts)))
        own_ages = real_X[recipe_row_idx[r], AGE_COL]
        jittered_recipe = jitter_continuous(recipe, real_X, zero_cols, rng, jitter_std_ratio)
        for _ in range(n_rep):
            if len(rows) >= N:
                break
            row = jittered_recipe.copy()
            row[AGE_COL] = rng.choice(own_ages) if len(own_ages) > 1 else rng.choice(
                ages_global, p=age_probs_global)
            rows.append(row)
    return np.array(rows[:N])


# ============================================= [2] 정답 최적점 탐색 (RSM 수치최적화)
def find_optima_multi_restart(model, lo, hi, y_min, y_max, G, min_sep_ratio, seed,
                              n_restarts=200):
    """RSM 위에서 desirability를 최대화하는 지점을 다중 시작점으로 탐색,
    서로 충분히 떨어진(min_sep_ratio) 상위 G개를 정답으로 채택."""
    rng = np.random.default_rng(seed)
    span = hi - lo
    bounds = list(zip(lo, hi))

    def neg_D(x):
        y_pred = rsm_predict(model, x.reshape(1, -1))[0]
        return -desirability_from_strength(np.array([y_pred]), y_min, y_max)[0]

    found = []
    for _ in range(n_restarts):
        x0 = rng.uniform(lo, hi)
        res = minimize(neg_D, x0, method='L-BFGS-B', bounds=bounds)
        if res.success:
            found.append((res.x, -res.fun))

    found.sort(key=lambda r: -r[1])
    centers, vals = [], []
    for x, d in found:
        if all(np.sqrt(np.mean(((x - c) / span) ** 2)) >= min_sep_ratio for c in centers):
            centers.append(x)
            vals.append(d)
        if len(centers) >= G:
            break

    return np.array(centers), np.array(vals)


# ============================================= [3] 전체 합성 데이터 생성
def generate_synthetic_data(real_X, real_y, model, N, G, min_sep_ratio, seed):
    y_min, y_max = real_y.min(), real_y.max()
    lo, hi = real_X.min(axis=0), real_X.max(axis=0)
    span = hi - lo

    centers, center_D = find_optima_multi_restart(
        model, lo, hi, y_min, y_max, G, min_sep_ratio, seed)
    G_found = len(centers)   # RSM 지형상 실제로 찾아지는 국소최적점 수 (G보다 적을 수 있음)

    X = sample_recipe_replicate_inputs(real_X, N, seed)
    y_pred = rsm_predict(model, X)
    D = desirability_from_strength(y_pred, y_min, y_max)

    dist_to_centers = np.zeros((N, max(G_found, 1)))
    for g in range(G_found):
        dist_to_centers[:, g] = np.sqrt(np.mean(((X - centers[g]) / span) ** 2, axis=1))
    truth_sets = [set(np.where(dist_to_centers[:, g] <= MATCH_RADIUS_RATIO)[0].tolist())
                 for g in range(G_found)]

    return dict(X=X, D=D, centers=centers, center_D=center_D, G_found=G_found,
               truth_sets=truth_sets, y_pred=y_pred)


# ============================================= 파이프라인 (원논문 방식, 이전과 동일)
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


def score_against_truth(boxes, lab, n_memb_star, truth_sets):
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
    pairs = []
    for ci, C in enumerate(eff_clusters):
        for gi, T in enumerate(truth_sets):
            inter = len(C & T)
            prec = inter / len(C) if C else 0.0
            rec = inter / len(T) if T else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pairs.append((f1, ci, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_c, matched_g, results = set(), set(), []
    for f1, ci, gi in pairs:
        if ci in matched_c or gi in matched_g:
            continue
        matched_c.add(ci); matched_g.add(gi)
        results.append((gi, ci, f1))

    mean_f1 = np.mean([r[2] for r in results]) if results else 0.0
    return dict(N_eff=N_eff, G=G, mean_f1=mean_f1, n_matched=len(results))


# ============================================= 실행
def run_scenario(real_X, real_y, model, label, N=1200, G=3, min_sep_ratio=0.20,
                 min_support_ratio=0.10, seed=0, verbose=True):
    data = generate_synthetic_data(real_X, real_y, model, N, G,
                                   min_sep_ratio=min_sep_ratio, seed=seed)
    min_support = max(20, int(N * min_support_ratio))
    boxes = build_boxes(data['X'], data['D'], min_support, seed)
    if len(boxes) < 5:
        return dict(label=label, error='박스 생성 실패')

    Z, sets, M = build_linkage(boxes)
    final = full_grid_search(Z, sets, M)
    if final is None:
        return dict(label=label, error='rho_Limit 만족 K 없음')
    n_memb_star, K_star, N_eff, ecc, rho = final
    lab = fcluster(Z, K_star, criterion='maxclust')

    score = score_against_truth(boxes, lab, n_memb_star, data['truth_sets'])
    result = dict(label=label, G_target=G, G_found=data['G_found'],
                  N_eff=score['N_eff'], mean_f1=score['mean_f1'],
                  n_matched=score['n_matched'], K_star=K_star, ECC=ecc,
                  n_boxes=len(boxes))
    if verbose:
        print(f'  [{label}] G(목표)={G}, G(실제탐색)={data["G_found"]} -> '
              f'N_eff={score["N_eff"]}, 평균F1={score["mean_f1"]:.3f}, '
              f'매칭={score["n_matched"]}/{data["G_found"]}, '
              f'K*={K_star}, ECC={ecc:.3f}, 박스={len(boxes)}개')
    return result


def main():
    real_X, real_y = load_real_Xy(CSV_PATH)
    print(f'실제 데이터 {len(real_y)}개 로드\n')

    print('반응표면모델(RSM) 적합 중 (주효과+이차항+교호작용, OLS) ...')
    model = fit_response_surface(real_X, real_y)
    # ---------- 진단: RSM 지형과 최적점이 제대로 나오는지 확인 ----------
    print('=' * 78)
    print('  진단: RSM 예측값 분포 및 desirability 지형')
    print('=' * 78)
    y_pred_all = rsm_predict(model, real_X)
    print(f'  실제 데이터에서의 RSM 예측 강도: '
          f'min={y_pred_all.min():.1f}, max={y_pred_all.max():.1f}, '
          f'mean={y_pred_all.mean():.1f}')
    print(f'  실측 강도(real_y)          : '
          f'min={real_y.min():.1f}, max={real_y.max():.1f}, mean={real_y.mean():.1f}')
    print(f'  Target=60 이 실측 범위 안에 있는가? '
          f'{"예" if real_y.min() <= 60 <= real_y.max() else "아니오"}')

    D_all = desirability_from_strength(y_pred_all, real_y.min(), real_y.max())
    print(f'  실제 데이터 전체의 desirability(D) 분포: '
          f'min={D_all.min():.3f}, max={D_all.max():.3f}, mean={D_all.mean():.3f}')
    print(f'  D>0.9인 실측 관측치 개수: {(D_all > 0.9).sum()} / {len(D_all)}\n')

    lo, hi = real_X.min(axis=0), real_X.max(axis=0)
    centers, center_D = find_optima_multi_restart(
        model, lo, hi, real_y.min(), real_y.max(), G=5, min_sep_ratio=0.20, seed=1)
    print(f'  탐색된 최적점 {len(centers)}개의 desirability 값:')
    for i, d in enumerate(center_D):
        print(f'    center {i}: D = {d:.4f}')
    print()
    print(f'  적합 완료: R^2 = {model["r2"]:.3f} '
          f'(이 회귀식을 desirability의 "실제 관계식"으로 사용)\n')

    print('=' * 78)
    print('  RSM 기반 정답 복원 검증')
    print('=' * 78)
    for G in [2, 3, 5]:
        run_scenario(real_X, real_y, model, f'G={G}', N=1200, G=G, seed=1)

    print('\n완료.')


if __name__ == '__main__':
    main()
