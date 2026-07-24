"""
================================================================================
합성 데이터 정답 기반 검증 — 3차 버전 (Smoothed Bootstrap + Recipe-Replicate)
================================================================================
2차 버전(Gaussian copula)의 한계
  실제 콘크리트 데이터를 뜯어본 결과, 코퓰라(단일 상관계수) 방식으로는
  재현되지 않는 구조적 특징이 세 가지 확인됨:
    1) 극심한 zero-inflation: Slag 45.7%, FlyAsh 55.0%, Superplast 36.8%가
       정확히 0이고, 이 셋이 "함께 0이 되는" 조합이 군집화되어 있음
       (배합 "레시피 계열"의 존재를 시사 — 단일 코퓰라로 재현 불가).
    2) Age는 사실상 범주형: 실제 값은 14개(1,3,7,...,365)뿐이고 28일이 41%.
       코퓰라의 quantile-interpolation은 존재하지 않는 재령(예: 45일)을 만들어냄.
    3) 레시피 반복측정 구조: 조성 7개 변수만 보면 1030행 중 고유 레시피는
       427개뿐 (평균 2.4회 반복측정). 코퓰라는 매번 완전히 새 조합을 만들어
       이 "중복/군집" 패턴을 재현하지 못함.

3차 개선
  D(정답 함수)는 그대로 유지 — 정답 위치를 통제할 수 있어야 검증 데이터로서
  의미가 있으므로, 실제 desirability 관계는 쓰지 않고 가우시안 봉우리를 그대로
  심는다. 개선 대상은 X(입력) 생성부뿐이다.
    - Smoothed bootstrap: 실제 행을 복원추출 + 작은 지터. 정확히 0인 값은
      지터에서 제외(0을 그대로 유지)하여 zero-inflation을 보존.
    - Age: 지터하지 않고 실제 14개 값 중에서 실제 빈도로 범주형 리샘플.
    - Recipe-replicate 2단계 샘플링(선택, 기본 사용): 먼저 고유 레시피(7개
      조성 변수 조합)를 실제 데이터에서 추출하고, 레시피당 반복측정 개수를
      실제 분포(평균 2.4회)에서 뽑아 그 개수만큼 복제한 뒤 Age/지터를 적용.
      이렇게 하면 "동일 배합 반복측정" 군집 패턴까지 보존된다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

FEAT_NAMES = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
              'CoarseAgg', 'FineAgg', 'Age']
AGE_COL = FEAT_NAMES.index('Age')
ZERO_INFLATION_THRESHOLD = 0.05   # 이 비율 이상 정확히 0이면 '제로팽창 변수'로 취급
JITTER_STD_RATIO = 0.03           # 지터 강도 (변수 표준편차 대비 비율)

# --- 파이프라인 하이퍼파라미터 (2차 버전과 동일 계열, 그대로 유지) ---
ALPHA_PEEL = 0.05
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20
S_RATIO_LOW, S_RATIO_HIGH = 0.5, 0.75
T_PER_SIZE = 300
MATCH_RADIUS_RATIO = 0.20


# ============================================= [1] 합성 데이터 생성 (smoothed bootstrap + recipe-replicate)
def load_real_X(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df.iloc[:, :8].values.astype(float)


def zero_inflated_columns(real_X, threshold=ZERO_INFLATION_THRESHOLD):
    """정확히 0인 값의 비율이 threshold를 넘는 열 인덱스(Age 제외)를 찾는다."""
    P = real_X.shape[1]
    return [p for p in range(P) if p != AGE_COL
            and (real_X[:, p] == 0).mean() >= threshold]


def jitter_continuous(row, real_X, zero_cols, rng, jitter_std_ratio=JITTER_STD_RATIO):
    """
    연속형 변수에 작은 가우시안 지터를 더한다.
    - 제로팽창 변수(zero_cols)는 정확히 0인 값이면 지터하지 않고 0 그대로 유지
      (0이 아닌 값만 지터).
    - Age(범주형)는 건드리지 않는다(호출부에서 별도 처리).
    - 지터 후 실제 관측 범위로 clip.
    """
    out = row.copy()
    stds = real_X.std(axis=0)
    lo, hi = real_X.min(axis=0), real_X.max(axis=0)
    P = len(row)
    for p in range(P):
        if p == AGE_COL:
            continue
        if p in zero_cols and row[p] == 0.0:
            continue
        out[p] = row[p] + rng.normal(0, jitter_std_ratio * stds[p])
        out[p] = np.clip(out[p], lo[p], hi[p])
    return out


def sample_smoothed_bootstrap_inputs(real_X, N, seed, jitter_std_ratio=JITTER_STD_RATIO):
    """
    단순 smoothed bootstrap (recipe 구조 없이): 실제 행을 복원추출 + 지터,
    Age는 실제 빈도 기반 범주형 리샘플로 교체.
    """
    rng = np.random.default_rng(seed)
    N_real, P = real_X.shape
    zero_cols = zero_inflated_columns(real_X)

    ages, age_counts = np.unique(real_X[:, AGE_COL], return_counts=True)
    age_probs = age_counts / age_counts.sum()

    rows_idx = rng.integers(0, N_real, size=N)
    X = np.zeros((N, P))
    for i, ridx in enumerate(rows_idx):
        X[i] = jitter_continuous(real_X[ridx], real_X, zero_cols, rng, jitter_std_ratio)
        X[i, AGE_COL] = rng.choice(ages, p=age_probs)
    return X


def build_recipe_pool(real_X):
    """조성 7개 변수(Age 제외) 기준 고유 레시피와, 레시피별 원본 행 인덱스 목록을 추출."""
    non_age_idx = [p for p in range(real_X.shape[1]) if p != AGE_COL]
    keys = [tuple(np.round(real_X[i, non_age_idx], 6)) for i in range(len(real_X))]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    recipes = [np.array(real_X[idxs[0]]) for idxs in groups.values()]
    recipe_row_idx = list(groups.values())
    return recipes, recipe_row_idx, non_age_idx


def sample_recipe_replicate_inputs(real_X, N, seed, jitter_std_ratio=JITTER_STD_RATIO):
    """
    2단계 샘플링:
      1) 실제 데이터에서 추출한 고유 레시피(427개 상당) 중 하나를 균등 추출
      2) 그 레시피의 반복측정 횟수를 실제 반복측정 개수 분포(평균 2.4회)에서 추출
      3) 반복 개수만큼 복제 -> 각 복제본에 지터 적용, Age는 그 레시피가 실제로
         관측된 Age들 중에서(또는 전역 Age 분포에서) 리샘플
    실제 데이터의 "동일 배합 반복측정" 군집 패턴을 보존하는 것이 목적.
    """
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
        n_rep = int(rng.choice(replicate_counts))
        n_rep = max(1, n_rep)
        # 이 레시피가 실제로 관측된 Age들 (반복측정마다 Age가 달라지는 실제 패턴 반영)
        own_ages = real_X[recipe_row_idx[r], AGE_COL]
        # 지터는 레시피 단위로 한 번만 적용 (실제 데이터처럼 같은 배치는 조성이
        # 완전히 동일하고, 재령만 달라지는 반복측정 구조를 보존하기 위함)
        jittered_recipe = jitter_continuous(recipe, real_X, zero_cols, rng, jitter_std_ratio)
        for _ in range(n_rep):
            if len(rows) >= N:
                break
            row = jittered_recipe.copy()
            if len(own_ages) > 1:
                row[AGE_COL] = rng.choice(own_ages)
            else:
                row[AGE_COL] = rng.choice(ages_global, p=age_probs_global)
            rows.append(row)
    return np.array(rows[:N])


def sample_centers(real_X, G, min_sep_ratio, seed):
    """정답 중심을 실제 관측치들 중에서 골라 배치 (실존하지 않는 값 조합 방지)"""
    rng = np.random.default_rng(seed)
    lo, hi = real_X.min(axis=0), real_X.max(axis=0)
    span = hi - lo
    N_real = len(real_X)

    centers, attempts = [], 0
    while len(centers) < G and attempts < 5000:
        cand = real_X[rng.integers(0, N_real)]
        if all(np.sqrt(np.mean(((cand - c) / span) ** 2)) >= min_sep_ratio
              for c in centers):
            centers.append(cand)
        attempts += 1
    if len(centers) < G:
        raise RuntimeError(f'min_sep_ratio={min_sep_ratio} 로 {G}개 중심 배치 실패')
    return np.array(centers)


def generate_synthetic_data(real_X, N, G, A, sigma_ratio, noise_std,
                            baseline, min_sep_ratio, seed, use_recipe_structure=True):
    P = real_X.shape[1]
    span = real_X.max(axis=0) - real_X.min(axis=0)

    centers = sample_centers(real_X, G, min_sep_ratio, seed)
    if use_recipe_structure:
        X = sample_recipe_replicate_inputs(real_X, N, seed)
    else:
        X = sample_smoothed_bootstrap_inputs(real_X, N, seed)

    D = np.full(N, baseline)
    dist_to_centers = np.zeros((N, G))
    for g in range(G):
        d_norm = np.sqrt(np.mean(((X - centers[g]) / span) ** 2, axis=1))
        dist_to_centers[:, g] = d_norm
        D += A[g] * np.exp(-(d_norm ** 2) / (2 * sigma_ratio ** 2))
    D += np.random.default_rng(seed + 999).normal(0, noise_std, size=N)
    D = np.clip(D, 0.0, 1.0)

    truth_sets = [set(np.where(dist_to_centers[:, g] <= MATCH_RADIUS_RATIO)[0].tolist())
                 for g in range(G)]

    return dict(X=X, D=D, centers=centers, span=span, truth_sets=truth_sets,
               feat_names=FEAT_NAMES[:P])


# ============================================= [1-검증] 합성 데이터가 실제 구조 특징을 보존하는지 확인
def validate_structure(real_X, X_synth, label):
    print(f'  --- 구조 보존 검증 [{label}] ---')
    for p, name in enumerate(FEAT_NAMES[:7]):
        real_zf = (real_X[:, p] == 0).mean()
        synth_zf = (X_synth[:, p] == 0).mean()
        print(f'    {name:10s} zero-frac 실제={real_zf:.3f} 합성={synth_zf:.3f}')
    real_ages = set(np.unique(real_X[:, AGE_COL]).tolist())
    synth_ages = set(np.unique(X_synth[:, AGE_COL]).tolist())
    print(f'    Age 고유값: 실제={len(real_ages)}개, 합성={len(synth_ages)}개, '
          f'합성값이 실제값의 부분집합={synth_ages.issubset(real_ages)}')
    _, real_recipe_idx, _ = build_recipe_pool(real_X)
    _, synth_recipe_idx, _ = build_recipe_pool(X_synth)
    print(f'    고유 레시피 수: 실제={len(real_recipe_idx)}/{len(real_X)}행, '
          f'합성={len(synth_recipe_idx)}/{len(X_synth)}행')


# ============================================= [2] 기존 파이프라인 (MRS-PRIM + 클러스터링) — 2차 버전과 동일
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


# ============================================= [3] 채점 — 2차 버전과 동일
def score_against_truth(boxes, lab, n_memb_star, truth_sets, N_total):
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
def run_scenario(real_X, label, N=1000, G=3, A=None, sigma_ratio=0.15,
                 noise_std=0.05, min_sep_ratio=0.15, min_support_ratio=0.10,
                 seed=0, use_recipe_structure=True, verbose=True):
    if A is None:
        A = [0.7] * G
    data = generate_synthetic_data(real_X, N, G, A, sigma_ratio, noise_std,
                                   baseline=0.05, min_sep_ratio=min_sep_ratio,
                                   seed=seed, use_recipe_structure=use_recipe_structure)
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


# ============================================= 시나리오 반복 실행 (무작위 시드 + 평균/표준편차 요약)
def run_scenario_repeated(real_X, label, n_repeats=5, **scenario_kwargs):
    """
    같은 시나리오 설정(G, noise 등)을 매 반복마다 완전히 무작위인 시드로
    재실행하여, 원 논문 방법론이 얼마나 안정적으로 정답을 복원하는지
    평균/표준편차로 요약한다. 사용한 시드도 함께 기록해 필요하면 특정
    반복(예: 이상치)만 재현해볼 수 있게 한다.
    """
    scenario_kwargs.pop('seed', None)
    scenario_kwargs.pop('verbose', None)
    seed_rng = np.random.default_rng()  # 매 실행마다 다른 entropy로 초기화 (완전 무작위)

    runs = []
    for _ in range(n_repeats):
        seed = int(seed_rng.integers(0, 2**31 - 1))
        r = run_scenario(real_X, label, seed=seed, verbose=False, **scenario_kwargs)
        r['seed'] = seed
        runs.append(r)

    valid = [r for r in runs if 'error' not in r]
    failed = n_repeats - len(valid)
    if not valid:
        print(f'  [{label}] 전체 {n_repeats}회 모두 실패 (박스/클러스터 생성 불가)')
        return dict(label=label, n_repeats=n_repeats, n_failed=failed, runs=runs)

    f1s = np.array([r['mean_f1'] for r in valid])
    count_matches = np.array([r['count_match'] for r in valid])
    n_effs = np.array([r['N_eff'] for r in valid])

    summary = dict(
        label=label, n_repeats=n_repeats, n_failed=failed,
        f1_mean=f1s.mean(), f1_std=f1s.std(),
        count_match_rate=count_matches.mean(),
        n_eff_mean=n_effs.mean(), n_eff_std=n_effs.std(),
        seeds=[r['seed'] for r in runs], runs=runs,
    )
    print(f'  [{label}] {len(valid)}/{n_repeats}회 성공 (실패 {failed}회) | '
          f'F1={summary["f1_mean"]:.3f}±{summary["f1_std"]:.3f} | '
          f'개수일치율={summary["count_match_rate"]*100:.0f}% | '
          f'N_eff={summary["n_eff_mean"]:.1f}±{summary["n_eff_std"]:.1f} | '
          f'seeds={summary["seeds"]}')
    return summary


# ============================================= 난이도별 반복 실행
def main(n_repeats=5):
    real_X = load_real_X(CSV_PATH)
    print(f'실제 관측치 {len(real_X)}개 로드 완료\n')

    print('=' * 78)
    print('  [0] 합성 X가 실제 구조적 특징을 보존하는지 확인')
    print('=' * 78)
    X_boot = sample_smoothed_bootstrap_inputs(real_X, N=1200, seed=0)
    validate_structure(real_X, X_boot, 'smoothed bootstrap (레시피 구조 없음)')
    X_recipe = sample_recipe_replicate_inputs(real_X, N=1200, seed=0)
    validate_structure(real_X, X_recipe, 'recipe-replicate 2단계 샘플링')

    print('\n' + '=' * 78)
    print(f'  난이도별 정답 복원 검증 (recipe-replicate 구조 사용, 시나리오당 무작위 시드 {n_repeats}회 반복)')
    print('=' * 78)

    summaries = []

    print('\n[A] 정답 개수(G) 변화 (거리/noise 고정)')
    for G in [2, 3, 5]:
        s = run_scenario_repeated(real_X, f'G={G}', n_repeats=n_repeats, N=1200, G=G,
                                  A=[0.7]*G, sigma_ratio=0.15, noise_std=0.05,
                                  min_sep_ratio=0.20)
        summaries.append(s)

    print('\n[B] 중심 간 거리(min_sep_ratio) 변화 (G=3 고정)')
    for sep in [0.10, 0.20, 0.35]:
        s = run_scenario_repeated(real_X, f'sep={sep}', n_repeats=n_repeats, N=1200, G=3,
                                  A=[0.7]*3, sigma_ratio=0.15, noise_std=0.05,
                                  min_sep_ratio=sep)
        summaries.append(s)

    print('\n[C] noise 수준 변화 (G=3 고정)')
    for noise in [0.02, 0.05, 0.10]:
        s = run_scenario_repeated(real_X, f'noise={noise}', n_repeats=n_repeats, N=1200, G=3,
                                  A=[0.7]*3, sigma_ratio=0.15, noise_std=noise,
                                  min_sep_ratio=0.20)
        summaries.append(s)

    print('\n[D] 상대적 세기(A) 불균형 (G=3, 약한 영역 포함)')
    for A in ([0.7, 0.7, 0.7], [0.9, 0.5, 0.2]):
        s = run_scenario_repeated(real_X, f'A={A}', n_repeats=n_repeats, N=1200, G=3,
                                  A=A, sigma_ratio=0.15, noise_std=0.05,
                                  min_sep_ratio=0.20)
        summaries.append(s)

    print('\n' + '=' * 78)
    print('  종합 요약 (시나리오별 평균 F1 / 개수일치율의 전체 평균)')
    print('=' * 78)
    valid = [s for s in summaries if 'f1_mean' in s]
    if valid:
        print(f'  전체 시나리오 평균 F1: {np.mean([s["f1_mean"] for s in valid]):.3f}')
        print(f'  전체 시나리오 평균 개수일치율: {np.mean([s["count_match_rate"] for s in valid])*100:.0f}%')

    print('\n완료.')


if __name__ == '__main__':
    main()
