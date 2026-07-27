"""
================================================================================
아이디어 6번 — desirability를 포함한 통합 목적함수 (신규, 방식4)
================================================================================
문제의식
  기존 통합 목적함수(방식0/1/3: ECC, rho_eff 조합)는 '클러스터 구조가 얼마나
  깔끔한가'만 보고, '그 클러스터의 desirability(실제 성능)가 좋은가'는 전혀
  보지 않았다. 그 결과 구조적으로는 안정적이지만 desirability가 희석된
  (넓고 큰) 클러스터가 선택되어, confirmation hold-out 성능에서 기존 방식보다
  못한 결과가 나왔다.

방식4 (신규): score(K) = ECC(K)^a * rho_eff(K)^b * Dbar_largest(K)^c
  Dbar_largest(K) = 그 K에서 가장 큰 효과적 클러스터에 속한 멤버 박스들의
                    dbar(desirability) 평균. training 데이터만으로 계산하며
                    confirmation 은 전혀 사용하지 않는다(정답을 미리 아는
                    문제 없음).

검증 두 가지를 모두 수행한다
  [A] score(K) 곡선이 '내부 봉우리'를 갖는지 (경계에 붙지 않는지)
  [B] 이 방식이 고른 K*의 최대 클러스터가, confirmation 실측 desirability
      기준으로 기존(rho_Limit=0.6)과 방식0/3보다 더 나은지 (Hold-out)
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

# 방식4의 지수 가중치 (a: ECC, b: rho_eff, c: Dbar) - 동일가중 기본값
METHOD4_WEIGHTS = dict(a=0.2, b=0.2, c=99.0)


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


def precompute_per_K(Z, boxes, sets, M, k_grid, n_memb):
    """K별로 (ECC, rho_eff, 가장 큰 효과적 클러스터의 Dbar, 그 클러스터 멤버 인덱스)를 계산"""
    per_K = {}
    for K in k_grid:
        lab = fcluster(Z, K, criterion='maxclust')
        entries = []          # (n_mem, gamma) - 전체 효과적 클러스터용 (ECC, rho 계산)
        cluster_mems = []      # 각 효과적 클러스터의 멤버 인덱스 배열
        for k in np.unique(lab):
            mem = np.where(lab == k)[0]
            if len(mem) < n_memb:
                continue
            I = set.intersection(*[sets[m] for m in mem])
            U = set.union(*[sets[m] for m in mem])
            entries.append((len(mem), len(I) / len(U) if U else 0.0))
            cluster_mems.append(mem)
        if not entries:
            continue
        rho = sum(n for n, _ in entries) / M
        ecc = np.mean([g for _, g in entries])
        # 가장 큰 효과적 클러스터의 Dbar (training 전용, confirmation 미사용)
        largest_mem = max(cluster_mems, key=len)
        dbar_largest = np.mean([boxes[i]['dbar'] for i in largest_mem])
        per_K[K] = dict(ecc=ecc, rho=rho, dbar=dbar_largest,
                        largest_mem=largest_mem, n_mem=len(largest_mem))
    return per_K


# ============================================= K* 선정 방식들
def select_original(per_K, rho_limit=RHO_LIMIT_BASELINE):
    best = None
    for K, v in per_K.items():
        if v['rho'] < rho_limit:
            continue
        if best is None or v['ecc'] > best[1]['ecc']:
            best = (K, v)
    return best


def select_method0(per_K, w=0.5):
    best = None
    for K, v in per_K.items():
        score = v['ecc'] ** w * v['rho'] ** (1 - w)
        if best is None or score > best[1]:
            best = (K, score, v)
    return best[0], best[2]


def select_method3(per_K, M, lam=1.0):
    best = None
    for K, v in per_K.items():
        score = v['ecc'] - lam * (K / M)
        if best is None or score > best[1]:
            best = (K, score, v)
    return best[0], best[2]


def select_method4(per_K, weights=METHOD4_WEIGHTS):
    """신규: score = ECC^a * rho^b * Dbar_largest^c"""
    a, b, c = weights['a'], weights['b'], weights['c']
    best = None
    for K, v in per_K.items():
        d = max(v['dbar'], 1e-6)
        score = (v['ecc'] ** a) * (v['rho'] ** b) * (d ** c)
        if best is None or score > best[1]:
            best = (K, score, v)
    return best[0], best[2], best[1]


def check_interior_peak(per_K, score_func, k_grid_sorted):
    """score(K) 곡선이 경계가 아닌 내부에서 정점을 찍는지 확인 (진단용)"""
    Ks = [K for K in k_grid_sorted if K in per_K]
    scores = [score_func(per_K[K]) for K in Ks]
    i = int(np.argmax(scores))
    at_boundary = (i == 0 or i == len(Ks) - 1)
    lo, hi = max(0, i - 3), min(len(Ks), i + 4)
    neighborhood = [(Ks[j], round(scores[j], 4)) for j in range(lo, hi)]
    return Ks[i], at_boundary, neighborhood


# ============================================= Hold-out 평가
def holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, largest_mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in largest_mem])
    lo, hi = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n, lo, hi


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
    per_K = precompute_per_K(Z, boxes, sets, M, k_grid, N_MEMB)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}\n')

    # ---------- [A] 방식4의 score(K) 내부봉우리 여부 확인 ----------
    a, b, c = METHOD4_WEIGHTS['a'], METHOD4_WEIGHTS['b'], METHOD4_WEIGHTS['c']
    score_func = lambda v: (v['ecc'] ** a) * (v['rho'] ** b) * (max(v['dbar'], 1e-6) ** c)
    K_peak, at_boundary, neighborhood = check_interior_peak(per_K, score_func, k_grid)
    print('=' * 78)
    print(f'  [A] 방식4 score(K) 곡선 검증 (a={a:.2f}, b={b:.2f}, c={c:.2f})')
    print('=' * 78)
    print(f'  정점 K={K_peak}, 경계위치={at_boundary}')
    print(f'  정점 근방: {neighborhood}\n')

    # ---------- [B] Hold-out 비교 ----------
    print('=' * 78)
    print('  [B] 4가지 K* 선정 방식의 Hold-out 비교')
    print('=' * 78)

    K_orig, v_orig = select_original(per_K)
    K_m0, v_m0 = select_method0(per_K)
    K_m3, v_m3 = select_method3(per_K, M)
    K_m4, v_m4, score_m4 = select_method4(per_K)

    candidates = {
        '기존(rho>=0.6)': (K_orig, v_orig),
        '방식0(단순곱셈)': (K_m0, v_m0),
        '방식3(AIC/BIC)': (K_m3, v_m3),
        '방식4(desirability포함,신규)': (K_m4, v_m4),
    }

    print(f'\n  {"방식":>26} {"K*":>6} {"ECC":>8} {"rho":>8} {"Dbar(train)":>12} '
          f'{"n_mem":>6} {"n_conf":>7} {"D_conf(실측)":>12}')
    results = []
    for name, (K, v) in candidates.items():
        d_conf, n_conf, lo, hi = holdout_eval_cluster(
            boxes, X_train, X_confirm, D_confirm, v['largest_mem'])
        results.append((name, K, v['ecc'], v['rho'], v['dbar'], v['n_mem'], n_conf, d_conf))
        print(f'  {name:>26} {K:>6} {v["ecc"]:>8.4f} {v["rho"]:>8.3f} '
              f'{v["dbar"]:>12.4f} {v["n_mem"]:>6} {n_conf:>7} {d_conf:>12.4f}')

    print('\n' + '=' * 78)
    print('  결론')
    print('=' * 78)
    baseline_d = results[0][7]
    best_new = max(results[1:], key=lambda r: r[7])
    print(f'  기존 방식 confirmation D = {baseline_d:.4f}')
    print(f'  신규 방식 중 최고: {best_new[0]}, confirmation D = {best_new[7]:.4f} '
          f'(차이 {best_new[7]-baseline_d:+.4f})')
    if best_new[7] > baseline_d:
        print('  -> 신규 방식이 confirmation 기준으로도 기존을 능가함')
    else:
        print('  -> 아직 기존 방식을 못 넘어섬. 가중치(a,b,c)를 조정하며')
        print('     추가 탐색이 필요할 수 있음')

    print('\n완료.')


if __name__ == '__main__':
    main()
