"""
================================================================================
아이디어 6번 — N_Memb x K 완전탐색 (Factory 데이터)
================================================================================
콘크리트 버전과 동일한 논리를, factory 데이터(다중반응 15개)에 적용한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'continuous_factory_process.csv')

INPUT_COLS = [
    'AmbientConditions.AmbientHumidity.U.Actual',
    'AmbientConditions.AmbientTemperature.U.Actual',
    'Machine1.RawMaterial.Property1', 'Machine1.RawMaterial.Property2',
    'Machine1.RawMaterial.Property3', 'Machine1.RawMaterial.Property4',
    'Machine1.RawMaterialFeederParameter.U.Actual',
    'Machine1.Zone1Temperature.C.Actual', 'Machine1.Zone2Temperature.C.Actual',
    'Machine1.MotorAmperage.U.Actual', 'Machine1.MotorRPM.C.Actual',
    'Machine1.MaterialPressure.U.Actual', 'Machine1.MaterialTemperature.U.Actual',
    'Machine1.ExitZoneTemperature.C.Actual',
    'Machine2.RawMaterial.Property1', 'Machine2.RawMaterial.Property2',
    'Machine2.RawMaterial.Property3', 'Machine2.RawMaterial.Property4',
    'Machine2.RawMaterialFeederParameter.U.Actual',
    'Machine2.Zone1Temperature.C.Actual', 'Machine2.Zone2Temperature.C.Actual',
    'Machine2.MotorAmperage.U.Actual', 'Machine2.MotorRPM.C.Actual',
    'Machine2.MaterialPressure.U.Actual', 'Machine2.MaterialTemperature.U.Actual',
    'Machine2.ExitZoneTemperature.C.Actual',
    'Machine3.RawMaterial.Property1', 'Machine3.RawMaterial.Property2',
    'Machine3.RawMaterial.Property3', 'Machine3.RawMaterial.Property4',
    'Machine3.RawMaterialFeederParameter.U.Actual',
    'Machine3.Zone1Temperature.C.Actual', 'Machine3.Zone2Temperature.C.Actual',
    'Machine3.MotorAmperage.U.Actual', 'Machine3.MotorRPM.C.Actual',
    'Machine3.MaterialPressure.U.Actual', 'Machine3.MaterialTemperature.U.Actual',
    'Machine3.ExitZoneTemperature.C.Actual',
    'FirstStage.CombinerOperation.Temperature1.U.Actual',
    'FirstStage.CombinerOperation.Temperature2.U.Actual',
    'FirstStage.CombinerOperation.Temperature3.C.Actual',
]
N_RESPONSES = 15

TRAIN_RATIO = 0.7
SPLIT_SEED = 0

ALPHA_PEEL = 0.05
MIN_SUPPORT = 250
S_OPTIONS = (22, 27, 33)
T_PER_SIZE = 250
SEED_PRIM = 1
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]

METHOD4_ABC = dict(a=0.2, b=0.2, c=30.0)   # 콘크리트와 동일값(비교 기준), 필요시 조정


def build_multi_desirability(df, idx=None):
    d_list, targets = [], []
    for i in range(N_RESPONSES):
        y_all = df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)
        sp = df[f'Stage1.Output.Measurement{i}.U.Setpoint']
        nonzero = sp[sp != 0]
        target = float(nonzero.mode().iloc[0]) if len(nonzero) else float(sp.mean())
        lower, upper = float(y_all.min()), float(y_all.max())
        if upper <= target or target <= lower:
            lower, upper, target = y_all.min(), y_all.max(), float(np.median(y_all))
        targets.append((target, lower, upper))
        y = y_all if idx is None else y_all[idx]
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
        d_list.append(np.clip(d, 1e-6, 1.0))
    D_mat = np.column_stack(d_list)
    D_agg = np.exp(np.mean(np.log(D_mat), axis=1))
    return D_agg, targets


def desirability_from_targets(df, idx, targets):
    d_list = []
    for i, (target, lower, upper) in enumerate(targets):
        y = df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)[idx]
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
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


def build_boxes(X, D):
    rng = np.random.default_rng(SEED_PRIM)
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 100 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
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
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('continuous_factory_process.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH)
    X_all = df[INPUT_COLS].values.astype(float)
    n = len(df)
    print(f'데이터 {n}행, 입력 {X_all.shape[1]}개, 반응 {N_RESPONSES}개(다중반응)\n')

    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train = X_all[train_idx]
    X_confirm = X_all[confirm_idx]
    print(f'[0] 분할: training {len(train_idx)}개 / confirmation {len(confirm_idx)}개')

    D_train, targets = build_multi_desirability(df, idx=train_idx)
    D_confirm = desirability_from_targets(df, confirm_idx, targets)

    print('\n[1] training 데이터로 MRS-PRIM 박스 생성')
    boxes = build_boxes(X_train, D_train)
    print(f'  박스 {len(boxes)}개\n')

    print('클러스터링 및 K별 원본 클러스터 계산 ...')
    Z, sets, M = build_linkage(boxes, len(train_idx))
    k_grid = list(range(max(int(M * 0.05), 5), int(M * 0.50) + 1, K_GRID_STEP))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}, N_Memb 후보 {N_MEMB_OPTIONS}\n')

    print('=' * 90)
    print('  4가지 방식의 (N_Memb x K) 완전탐색 -> Hold-out 비교 (Factory)')
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
