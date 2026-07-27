"""
================================================================================
아이디어 6번 — Factory 데이터에서 원논문 vs 신규 방식 Hold-out 비교
================================================================================
콘크리트 하나만으로는 "우연히 원논문이 유리한 경우"인지 알 수 없으므로,
성격이 전혀 다른 factory 데이터(14088행, 41변수, 다중반응 15개)에서
동일한 비교(기존 rho>=0.6 vs 방식0/3/4)를 반복한다.
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
N_MEMB = 5
K_GRID_STEP = 5
RHO_LIMIT_BASELINE = 0.60

C_GRID = [0.2, 30, 99]


# ============================================= 다중반응 desirability
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


# ============================================= MRS-PRIM 박스 생성
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


# ============================================= 클러스터링 (벡터화, factory 규모 대응)
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


def precompute_per_K(Z, boxes, sets, M, k_grid, n_memb):
    per_K = {}
    for K in k_grid:
        lab = fcluster(Z, K, criterion='maxclust')
        entries, cluster_mems = [], []
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
        largest_mem = max(cluster_mems, key=len)
        dbar_largest = np.mean([boxes[i]['dbar'] for i in largest_mem])
        per_K[K] = dict(ecc=ecc, rho=rho, dbar=dbar_largest, largest_mem=largest_mem)
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


def select_method4(per_K, a, b, c):
    best = None
    for K, v in per_K.items():
        d = max(v['dbar'], 1e-6)
        score = (v['ecc'] ** a) * (v['rho'] ** b) * (d ** c)
        if best is None or score > best[1]:
            best = (K, score, v)
    return best[0], best[2]


# ============================================= Hold-out
def holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm, largest_mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in largest_mem])
    lo, hi = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


# ============================================= 실행
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

    print('클러스터링(벡터화) ...')
    Z, sets, M = build_linkage(boxes, len(train_idx))
    k_grid = list(range(max(int(M * 0.05), 5), int(M * 0.50) + 1, K_GRID_STEP))
    per_K = precompute_per_K(Z, boxes, sets, M, k_grid, N_MEMB)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}\n')

    print('=' * 80)
    print('  기존 vs 신규 방식 Hold-out 비교 (Factory 데이터)')
    print('=' * 80)

    candidates = {}
    orig = select_original(per_K)
    if orig is not None:
        candidates['기존(rho>=0.6)'] = orig
    candidates['방식0(단순곱셈)'] = select_method0(per_K)
    candidates['방식3(AIC/BIC)'] = select_method3(per_K, M)
    for c in C_GRID:
        candidates[f'방식4(c={c})'] = select_method4(per_K, 0.2, 0.2, c)

    print(f'\n  {"방식":>20} {"K*":>6} {"ECC":>8} {"rho":>8} {"Dbar":>8} '
          f'{"n_conf":>7} {"D_conf(실측)":>12}')
    results = []
    for name, (K, v) in candidates.items():
        d_conf, n_conf = holdout_eval_cluster(boxes, X_train, X_confirm, D_confirm,
                                              v['largest_mem'])
        results.append((name, K, v['ecc'], v['rho'], v['dbar'], n_conf, d_conf))
        print(f'  {name:>20} {K:>6} {v["ecc"]:>8.4f} {v["rho"]:>8.3f} '
              f'{v["dbar"]:>8.4f} {n_conf:>7} {d_conf:>12.4f}')

    print('\n' + '=' * 80)
    print('  결론')
    print('=' * 80)
    if '기존(rho>=0.6)' in candidates:
        baseline = [r for r in results if r[0] == '기존(rho>=0.6)'][0][6]
        others = [r for r in results if r[0] != '기존(rho>=0.6)']
        best_new = max(others, key=lambda r: r[6])
        print(f'  기존 방식 confirmation D = {baseline:.4f}')
        print(f'  신규 방식 중 최고: {best_new[0]}, D = {best_new[6]:.4f} '
              f'(차이 {best_new[6]-baseline:+.4f})')
        if best_new[6] >= baseline:
            print('  -> Factory 데이터에서는 신규 방식이 기존과 동등하거나 능가함')
        else:
            print('  -> Factory 데이터에서도 기존 방식을 못 넘어섬')
    else:
        print('  기존 방식이 rho_Limit=0.6 을 만족하는 K를 찾지 못함')

    print('\n완료.')


if __name__ == '__main__':
    main()
