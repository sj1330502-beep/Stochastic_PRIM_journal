"""
================================================================================
방식4의 (a,b,c) 자동탐색 — confirmation 전혀 사용하지 않음
================================================================================
sub-train/sub-val 처럼 데이터를 쪼개지 않고, training 데이터 자체를 그대로
'여러 번 재사용'한다. MRS-PRIM 박스 생성이 stochastic 이라는 점을 이용해:

  각 (a,b,c) 후보마다:
    1. 같은 training 데이터로 박스를 R번(다른 시드) 재생성
    2. 매번 방식4로 K* 를 고르고, 그 최대 클러스터의 (training 내) Dbar 기록
    3. R번의 평균 Dbar 와 (표준편차로 잰) 일관성을 계산

  score(a,b,c) = 평균(Dbar) - penalty * 표준편차(Dbar)

confirmation 은 전혀 쓰지 않으며, 최종 채택된 (a,b,c)를 가지고 딱 한 번만
(1) training 전체로 최종 박스를 만들고 (2) confirmation 으로 '결과 보고용'
채점을 한다 -- 이 마지막 단계는 파라미터 선택에 관여하지 않는, 순수한
최종 성적표 확인이다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'naval_propulsion_dataset.csv')

CONST_COLS = ['GT_Compressor_inlet_air_temp_T1', 'GT_Compressor_inlet_air_pressure_P1']
RESPONSE_COLS = ['GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient']

TRAIN_RATIO = 0.7
SPLIT_SEED = 0

ALPHA_PEEL = 0.05
MIN_SUPPORT = 250
S_OPTIONS = (7, 9, 11)
T_PER_SIZE = 150
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]

A_GRID = [0.05, 0.5, 1.0, 2.0]    # 훨씬 넓은 범위 (ECC가 다시 힘을 받는 지점까지)
B_GRID = [0.05, 0.5, 1.0, 2.0]
C_GRID = [0.5, 1, 5, 100]          # c를 아주 작게(desirability 거의 무시)도 포함
R_REPEATS = 3
STABILITY_PENALTY = 1.0


def build_multi_desirability_ltb(y_all_dict, idx=None):
    d_list, ranges = [], {}
    for name, y_all in y_all_dict.items():
        y_min, y_max = float(y_all.min()), float(y_all.max())
        ranges[name] = (y_min, y_max)
        y = y_all if idx is None else y_all[idx]
        d = (y - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y)
        d_list.append(np.clip(d, 1e-6, 1.0))
    D_mat = np.column_stack(d_list)
    return np.exp(np.mean(np.log(D_mat), axis=1)), ranges


def desirability_from_ranges(y_dict_idx, ranges):
    d_list = []
    for name, y in y_dict_idx.items():
        y_min, y_max = ranges[name]
        d = (y - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y)
        d_list.append(np.clip(d, 1e-6, 1.0))
    D_mat = np.column_stack(d_list)
    return np.exp(np.mean(np.log(D_mat), axis=1))


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


def build_linkage(boxes, N):
    M = len(boxes)
    Mem = np.zeros((M, N), dtype=np.int32)
    for i, b in enumerate(boxes):
        Mem[i, b['idx']] = 1
    inter = (Mem @ Mem.T).astype(float)
    sup = np.array([b['support'] for b in boxes], dtype=float)
    sim = inter / (sup[:, None] + sup[None, :] - inter)
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    sets = [set(b['idx'].tolist()) for b in boxes]
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


def cluster_range(boxes, X_train, mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    return X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)


def eval_range_on_confirm(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


def main():
    df = pd.read_csv(CSV_PATH)
    input_cols = [c for c in df.columns if c not in CONST_COLS + RESPONSE_COLS]
    X_all = df[input_cols].values.astype(float)
    n = len(df)
    print(f'데이터 {n}행, 입력 {len(input_cols)}개, 반응 {len(RESPONSE_COLS)}개(LTB)\n')

    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train = X_all[train_idx]
    X_confirm = X_all[confirm_idx]
    print(f'[0] 분할: training {len(train_idx)}개 / confirmation {len(confirm_idx)}개 '
          f'(confirmation은 마지막 보고에만 사용)\n')

    y_all_dict = {name: df[name].values.astype(float) for name in RESPONSE_COLS}
    D_train, ranges = build_multi_desirability_ltb(y_all_dict, idx=train_idx)
    y_confirm_dict = {name: y_all_dict[name][confirm_idx] for name in RESPONSE_COLS}
    D_confirm = desirability_from_ranges(y_confirm_dict, ranges)

    print('=' * 90)
    print(f'  1단계: training 데이터로 박스를 {R_REPEATS}회만 독립 생성 (재사용)')
    print('=' * 90)
    precomputed_raws = []   # [(raw, M, k_grid), ...] 길이 R_REPEATS
    for r in range(R_REPEATS):
        seed = 1000 + r
        print(f'  [반복 {r+1}/{R_REPEATS}] 박스 생성 중 (시드={seed}) ...')
        boxes_r = build_boxes(X_train, D_train, seed)
        Z_r, sets_r, M_r = build_linkage(boxes_r, len(train_idx))
        k_grid_r = list(range(max(int(M_r * 0.05), 5), int(M_r * 0.50) + 1, K_GRID_STEP))
        raw_r = precompute_raw_clusters(Z_r, sets_r, boxes_r, M_r, k_grid_r)
        precomputed_raws.append((raw_r, M_r, k_grid_r, boxes_r))
        print(f'    박스 {len(boxes_r)}개 생성 완료')

    print('\n' + '=' * 90)
    print(f'  2단계: (a,b,c) {len(A_GRID)}x{len(B_GRID)}x{len(C_GRID)}개 조합을 '
          f'미리 만든 {R_REPEATS}개 박스 결과에 재사용해 평가 (재생성 없음, 빠름)')
    print('=' * 90)

    grid_results = []
    for a in A_GRID:
        for b in B_GRID:
            for c in C_GRID:
                dbars = []
                for raw_r, M_r, k_grid_r, _ in precomputed_raws:
                    res = full_search_method4(raw_r, M_r, k_grid_r, N_MEMB_OPTIONS, a, b, c)
                    if res is not None:
                        dbars.append(res[2]['dbar'])
                if len(dbars) < 2:
                    continue
                mean_d, std_d = np.mean(dbars), np.std(dbars)
                score = mean_d - STABILITY_PENALTY * std_d
                grid_results.append((a, b, c, mean_d, std_d, score))
                print(f'  a={a}, b={b}, c={c:>3} | 평균Dbar(train)={mean_d:.4f} '
                      f'표준편차={std_d:.4f} score={score:.4f}')

    grid_results.sort(key=lambda r: -r[5])
    best = grid_results[0]
    a_star, b_star, c_star = best[0], best[1], best[2]
    print(f'\n  ▶ 채택(confirmation 미사용): a={a_star}, b={b_star}, c={c_star} '
          f'(평균Dbar={best[3]:.4f}, 표준편차={best[4]:.4f})')

    # 최종 보고: 이미 만들어둔 박스 결과 중 첫 번째를 그대로 재사용 (추가 생성 없음)
    print('\n[최종] 이미 생성된 박스(반복1)를 재사용해 채택된 (a,b,c)와 기존 방식 confirmation 비교')
    raw_full, M_full, k_grid_full, boxes_full = precomputed_raws[0]

    res_m4 = full_search_method4(raw_full, M_full, k_grid_full, N_MEMB_OPTIONS, a_star, b_star, c_star)
    n_memb_4, K_4, v_4 = res_m4
    lo_4, hi_4 = cluster_range(boxes_full, X_train, v_4['largest_mem'])
    d_conf_4, n_conf_4 = eval_range_on_confirm(lo_4, hi_4, X_confirm, D_confirm)

    res_orig = full_search_original(raw_full, M_full, k_grid_full, N_MEMB_OPTIONS)
    n_memb_o, K_o, v_o = res_orig
    lo_o, hi_o = cluster_range(boxes_full, X_train, v_o['largest_mem'])
    d_conf_o, n_conf_o = eval_range_on_confirm(lo_o, hi_o, X_confirm, D_confirm)

    print('\n' + '=' * 90)
    print('  최종 결과 (confirmation, 딱 1회만 사용)')
    print('=' * 90)
    print(f'  기존(rho>=0.6)              : K*={K_o}, n_conf={n_conf_o}, D_conf={d_conf_o:.4f}')
    print(f'  방식4(a={a_star},b={b_star},c={c_star}, 자동탐색) : K*={K_4}, n_conf={n_conf_4}, D_conf={d_conf_4:.4f}')
    print(f'\n  차이(방식4-기존) = {d_conf_4-d_conf_o:+.4f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
