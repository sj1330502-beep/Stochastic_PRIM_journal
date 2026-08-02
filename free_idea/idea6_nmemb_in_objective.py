"""
================================================================================
N_Memb를 목적함수에 직접 포함한 방식 vs 원논문 (콘크리트 데이터)
================================================================================
지금까지 N_Memb은 항상 '바깥에서 하나씩 대입해보는 격자탐색' 대상이었을 뿐,
목적함수 수식 안에 변수로 들어간 적이 없었다. 그리고 지금까지의 모든 실험에서
N_Memb*가 거의 항상 격자의 최솟값(5)으로 나왔는데, 이는 K에서 봤던 '경계
문제'(ECC(K) 단조증가로 인한 경계해)와 같은 구조일 수 있다는 가설을 세웠다.

이번 코드는 두 가지를 동시에 확인한다.
  [A] N_Memb 격자를 훨씬 넓게(2~20) 열었을 때도 원논문이 계속 경계(=2)로
      쏠리는지 진단
  [B] 새로운 목적함수에 N_Memb을 명시적인 항으로 포함시켜(지수 e), 이 벌점/
      보상이 경계쏠림을 막고 confirmation 성능도 개선하는지 검증

  score(K, N_Memb) = ECC^a * rho^b * Dbar^c * N_Memb^e
    (e>0 이면 'N_Memb이 클수록=효과적 클러스터 기준이 엄격할수록' 유리해지는
     방향으로, 표본이 극단적으로 적은 클러스터에 대한 신뢰도 문제를 완화하려는
     의도. e는 필요시 조정 가능한 파라미터.)
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

N_MEMB_WIDE = list(range(2, 21))

METHOD4_ABC = dict(a=0.2, b=0.2, c=30.0)
N_MEMB_EXP_E = 0.2


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
    return dict(ecc=ecc, rho=rho, dbar=largest['dbar'], largest_mem=largest['mem'])


def full_search_original_wide(raw, M, k_grid, n_memb_options, rho_limit=RHO_LIMIT_BASELINE):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None or v['rho'] < rho_limit:
                continue
            if best is None or v['ecc'] > best[2]['ecc']:
                best = (n_memb, K, v)
    return best


def full_search_method4_with_nmemb(raw, M, k_grid, n_memb_options, a, b, c, e):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            v = eval_for_nmemb(raw[K], n_memb, M)
            if v is None:
                continue
            d = max(v['dbar'], 1e-6)
            score = (v['ecc'] ** a) * (v['rho'] ** b) * (d ** c) * (n_memb ** e)
            if best is None or score > best[3]:
                best = (n_memb, K, v, score)
    return best[:3] if best else None


def cluster_range(boxes, X_train, mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    return X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)


def eval_range_on_confirm(lo, hi, X_confirm, D_confirm):
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

    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}, '
          f'N_Memb 격자(넓게) {N_MEMB_WIDE[0]}~{N_MEMB_WIDE[-1]}\n')

    print('=' * 90)
    print('  [A] 원논문 -- N_Memb 격자를 넓게 열었을 때도 경계(=2)로 쏠리는가?')
    print('=' * 90)
    res_o = full_search_original_wide(raw, M, k_grid, N_MEMB_WIDE)
    n_memb_o, K_o, v_o = res_o
    lo_o, hi_o = cluster_range(boxes, X_train, v_o['largest_mem'])
    d_o, n_o = eval_range_on_confirm(lo_o, hi_o, X_confirm, D_confirm)
    print(f'\n  원논문(N_Memb 넓은 격자): N_Memb*={n_memb_o}, K*={K_o}, '
          f'D_conf={d_o:.4f} (n={n_o})')
    if n_memb_o == N_MEMB_WIDE[0]:
        print('  -> 격자 최솟값으로 쏠림 확인됨 (경계 문제 가설 지지)')
    else:
        print(f'  -> 경계가 아닌 내부값({n_memb_o})에서 최적 -- 가설 기각')

    print('\n' + '=' * 90)
    print(f'  [B] 신규(N_Memb 포함 목적함수, e={N_MEMB_EXP_E}) vs 원논문')
    print('=' * 90)
    res_new = full_search_method4_with_nmemb(raw, M, k_grid, N_MEMB_WIDE,
                                             a=METHOD4_ABC['a'], b=METHOD4_ABC['b'],
                                             c=METHOD4_ABC['c'], e=N_MEMB_EXP_E)
    n_memb_n, K_n, v_n = res_new
    lo_n, hi_n = cluster_range(boxes, X_train, v_n['largest_mem'])
    d_n, n_n = eval_range_on_confirm(lo_n, hi_n, X_confirm, D_confirm)
    print(f'\n  신규(N_Memb 포함): N_Memb*={n_memb_n}, K*={K_n}, D_conf={d_n:.4f} (n={n_n})')
    print(f'  원논문(참고, 위와 동일): N_Memb*={n_memb_o}, K*={K_o}, D_conf={d_o:.4f}')

    print('\n' + '=' * 90)
    print('  결론')
    print('=' * 90)
    diff = d_n - d_o
    print(f'  차이(신규-원논문) = {diff:+.4f}')
    if diff > 0:
        print('  -> N_Memb을 목적함수에 포함한 신규 방식이 원논문을 능가함')
    elif diff == 0:
        print('  -> 신규 방식이 원논문과 동률')
    else:
        print('  -> 신규 방식도 원논문을 넘지 못함')

    print('\n완료.')


if __name__ == '__main__':
    main()
