"""
================================================================================
L-curve(엘보우 탐지) 방식 vs 원논문 (콘크리트 데이터)
================================================================================
지금까지의 방식들은 전부 ECC, rho_eff, Dbar를 '어떤 계산식으로 섞을지'
고민했다. 이번 방식은 아예 섞지 않는다.

ECC(K)와 rho_eff(K)를 로그-로그 평면에 곡선으로 그리면(K가 커질수록
ECC는 오르고 rho는 내리는 트레이드오프 곡선), 이 곡선이 가장 급격하게
꺾이는 지점('무릎', elbow)이 존재한다. 이 지점은 'rho를 더 희생해도 ECC가
별로 안 오르기 시작하는 경계'로, 별도의 가중치·지수·문턱값(rho_Limit) 없이
곡선 자체의 모양(곡률)만으로 자연스럽게 정해진다.

수치해석/역문제 분야에서 오래 쓰인 'L-curve criterion'을 그대로 적용한다.
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


def effective_clusters(raw_clusters_at_K, n_memb):
    return [c for c in raw_clusters_at_K if c['n'] >= n_memb]


def largest_of(eff):
    return max(eff, key=lambda c: c['n'])


def eval_ecc_rho(eff, M):
    if not eff:
        return None
    rho = sum(c['n'] for c in eff) / M
    ecc = np.mean([c['gamma'] for c in eff])
    return dict(ecc=ecc, rho=rho, largest_mem=largest_of(eff)['mem'])


def cluster_range(boxes, X_train, mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    return X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)


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
                best = (n_memb, K, v)
    return best


# ============================================= L-curve(엘보우) 방식
def curve_ecc_rho(raw, M, k_grid, n_memb):
    Ks, rhos, eccs, mems = [], [], [], []
    for K in k_grid:
        eff = effective_clusters(raw[K], n_memb)
        v = eval_ecc_rho(eff, M)
        if v is None:
            continue
        Ks.append(K); rhos.append(v['rho']); eccs.append(v['ecc']); mems.append(v['largest_mem'])
    return np.array(Ks), np.array(rhos), np.array(eccs), mems


def find_elbow(rhos, eccs):
    """
    로그-로그 평면에서 곡률이 최대인 지점(index)을 찾는다.
    곡률 근사: 이산 좌표에서 1차/2차 미분을 중심차분으로 근사.
    """
    eps = 1e-6
    x = np.log(np.clip(rhos, eps, None))
    y = np.log(np.clip(eccs, eps, None))
    n = len(x)
    if n < 5:
        return int(np.argmax(eccs))

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2 + eps, 1.5)
    interior = curvature[1:-1]
    if len(interior) == 0:
        return int(np.argmax(eccs))
    i_elbow = int(np.argmax(interior)) + 1
    return i_elbow


def elbow_search(raw, M, k_grid, n_memb_options):
    """N_Memb 별로 엘보우 지점을 찾고, 그중 ECC가 가장 높은 엘보우를 최종 채택
    (원논문의 '여러 N_Memb 중 ECC 최댓값' 최종선택 논리와 구조적으로 동일하게 유지)"""
    best = None
    for n_memb in n_memb_options:
        Ks, rhos, eccs, mems = curve_ecc_rho(raw, M, k_grid, n_memb)
        if len(Ks) < 3:
            continue
        i_elbow = find_elbow(rhos, eccs)
        K_e, rho_e, ecc_e, mem_e = Ks[i_elbow], rhos[i_elbow], eccs[i_elbow], mems[i_elbow]
        if best is None or ecc_e > best[3]:
            best = (n_memb, K_e, mem_e, ecc_e, rho_e)
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

    print('=' * 90)
    print('  L-curve(엘보우) vs 원논문')
    print('=' * 90)

    res_o = full_search_original(raw, M, k_grid, N_MEMB_OPTIONS)
    n_memb_o, K_o, v_o = res_o
    lo_o, hi_o = cluster_range(boxes, X_train, v_o['largest_mem'])
    d_o, n_o = eval_range_on_confirm(lo_o, hi_o, X_confirm, D_confirm)
    print(f'\n  기존(rho>=0.6): N_Memb*={n_memb_o}, K*={K_o}, D_conf={d_o:.4f} (n={n_o})')

    res_e = elbow_search(raw, M, k_grid, N_MEMB_OPTIONS)
    n_memb_e, K_e, mem_e, ecc_e, rho_e = res_e
    lo_e, hi_e = cluster_range(boxes, X_train, mem_e)
    d_e, n_e = eval_range_on_confirm(lo_e, hi_e, X_confirm, D_confirm)
    print(f'  L-curve 엘보우: N_Memb*={n_memb_e}, K*={K_e}, D_conf={d_e:.4f} (n={n_e}) '
          f'[ECC={ecc_e:.4f}, rho={rho_e:.3f}]')

    print('\n' + '=' * 90)
    print('  결론')
    print('=' * 90)
    diff = d_e - d_o
    print(f'  차이(엘보우-기존) = {diff:+.4f}')
    if diff > 0:
        print('  -> L-curve 엘보우 방식이 기존을 능가함')
    elif diff == 0:
        print('  -> L-curve 엘보우 방식이 기존과 동률')
    else:
        print('  -> L-curve 엘보우 방식도 기존을 넘지 못함')

    print('\n완료.')


if __name__ == '__main__':
    main()
