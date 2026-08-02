"""
================================================================================
파레토 최적 vs 원논문 — 데이터셋 설정만 바꾸면 재사용 가능한 범용 코드
================================================================================
사용법: 아래 [설정] 구역의 DATASET_NAME 값만 'concrete' / 'factory' / 'naval'
       중 하나로 바꾸면, 그에 맞는 CSV/변수/desirability 설정이 자동 적용된다.
       새로운 데이터셋을 추가하려면 PRESETS 딕셔너리에 항목만 추가하면 된다.

공통 절차: training/confirmation 분할 -> MRS-PRIM 박스 생성 -> 클러스터링 ->
  [기존(rho_Limit 제약 + ECC 최대화)] vs [파레토 최적(ECC,rho,Dbar 세 지표
  중 아무에게도 지배당하지 않는 K들 중 Dbar 최댓값 채택, 파라미터 없음)]
  두 방식을 N_Memb x K 완전탐색으로 비교하고 confirmation으로 채점한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))

# ================================================================= [설정]
DATASET_NAME = 'concrete'   # 'concrete' / 'factory' / 'naval' 중 선택

PRESETS = {
    'concrete': dict(
        csv_path='concrete_dataset.csv',
        desirability_mode='NTB_SINGLE',
        input_cols=None,
        response_cols=['Strength'],
        response_col_index=8,
        ntb_target=60.0,
        const_cols=[],
        alpha_peel=0.05, min_support=100,
        s_options=(4, 5, 6), t_per_size=667, seed_prim=1,
        k_grid_step=5, rho_limit=0.60,
        n_memb_options=list(range(2, 21)),   # 경계쏠림 진단을 위해 넓게
        train_ratio=0.7, split_seed=0,
    ),
    'factory': dict(
        csv_path='continuous_factory_process.csv',
        desirability_mode='NTB_MULTI_AUTOTARGET',
        input_cols=[
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
        ],
        response_cols=[f'Stage1.Output.Measurement{i}.U.Actual' for i in range(15)],
        response_col_index=None,
        ntb_target=None,
        const_cols=[],
        alpha_peel=0.05, min_support=250,
        s_options=(22, 27, 33), t_per_size=250, seed_prim=1,
        k_grid_step=5, rho_limit=0.60,
        n_memb_options=list(range(2, 21)),
        train_ratio=0.7, split_seed=0,
    ),
    'naval': dict(
        csv_path='naval_propulsion_dataset.csv',
        desirability_mode='LTB_MULTI',
        input_cols=None,
        response_cols=['GT_Compressor_decay_state_coefficient',
                       'GT_Turbine_decay_state_coefficient'],
        response_col_index=None,
        ntb_target=None,
        const_cols=['GT_Compressor_inlet_air_temp_T1', 'GT_Compressor_inlet_air_pressure_P1'],
        alpha_peel=0.05, min_support=250,
        s_options=(7, 9, 11), t_per_size=250, seed_prim=1,
        k_grid_step=5, rho_limit=0.60,
        n_memb_options=list(range(2, 21)),
        train_ratio=0.7, split_seed=0,
    ),
}

CFG = PRESETS[DATASET_NAME]
CSV_PATH = os.path.join(HERE, CFG['csv_path'])


def build_desirability(df, cfg, idx=None):
    mode = cfg['desirability_mode']

    if mode == 'NTB_SINGLE':
        y_all = df.iloc[:, cfg['response_col_index']].values.astype(float)
        lower, upper, target = float(y_all.min()), float(y_all.max()), cfg['ntb_target']
        y = y_all if idx is None else y_all[idx]
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower)
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target)
        return np.clip(d, 0.0, 1.0), dict(mode=mode, lower=lower, upper=upper, target=target)

    elif mode == 'NTB_MULTI_AUTOTARGET':
        d_list, ranges = [], []
        for col in cfg['response_cols']:
            y_all = df[col].values.astype(float)
            sp_col = col.replace('.Actual', '.Setpoint')
            sp = df[sp_col] if sp_col in df.columns else None
            if sp is not None:
                nonzero = sp[sp != 0]
                target = float(nonzero.mode().iloc[0]) if len(nonzero) else float(sp.mean())
            else:
                target = float(np.median(y_all))
            lower, upper = float(y_all.min()), float(y_all.max())
            if upper <= target or target <= lower:
                lower, upper, target = y_all.min(), y_all.max(), float(np.median(y_all))
            ranges.append((col, target, lower, upper))
            y = y_all if idx is None else y_all[idx]
            d = np.zeros_like(y)
            m1 = (y >= lower) & (y <= target)
            d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
            m2 = (y > target) & (y <= upper)
            d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
            d_list.append(np.clip(d, 1e-6, 1.0))
        D_mat = np.column_stack(d_list)
        D_agg = np.exp(np.mean(np.log(D_mat), axis=1))
        return D_agg, dict(mode=mode, ranges=ranges)

    elif mode == 'LTB_MULTI':
        d_list, ranges = [], []
        for col in cfg['response_cols']:
            y_all = df[col].values.astype(float)
            y_min, y_max = float(y_all.min()), float(y_all.max())
            ranges.append((col, y_min, y_max))
            y = y_all if idx is None else y_all[idx]
            d = (y - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y)
            d_list.append(np.clip(d, 1e-6, 1.0))
        D_mat = np.column_stack(d_list)
        D_agg = np.exp(np.mean(np.log(D_mat), axis=1))
        return D_agg, dict(mode=mode, ranges=ranges)

    else:
        raise ValueError(f'알 수 없는 desirability_mode: {mode}')


def desirability_with_fixed_ranges(df, cfg, idx, fixed):
    mode = fixed['mode']
    if mode == 'NTB_SINGLE':
        y_all = df.iloc[:, cfg['response_col_index']].values.astype(float)
        y = y_all[idx]
        lower, upper, target = fixed['lower'], fixed['upper'], fixed['target']
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower)
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target)
        return np.clip(d, 0.0, 1.0)

    elif mode == 'NTB_MULTI_AUTOTARGET':
        d_list = []
        for col, target, lower, upper in fixed['ranges']:
            y = df[col].values.astype(float)[idx]
            d = np.zeros_like(y)
            m1 = (y >= lower) & (y <= target)
            d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
            m2 = (y > target) & (y <= upper)
            d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
            d_list.append(np.clip(d, 1e-6, 1.0))
        D_mat = np.column_stack(d_list)
        return np.exp(np.mean(np.log(D_mat), axis=1))

    elif mode == 'LTB_MULTI':
        d_list = []
        for col, y_min, y_max in fixed['ranges']:
            y = df[col].values.astype(float)[idx]
            d = (y - y_min) / (y_max - y_min) if y_max > y_min else np.ones_like(y)
            d_list.append(np.clip(d, 1e-6, 1.0))
        D_mat = np.column_stack(d_list)
        return np.exp(np.mean(np.log(D_mat), axis=1))


def get_input_matrix(df, cfg):
    if cfg['input_cols'] is not None:
        cols = cfg['input_cols']
    else:
        exclude = set(cfg['const_cols']) | set(cfg.get('response_cols') or [])
        if cfg['response_col_index'] is not None:
            all_cols = list(df.columns)
            cols = [c for i, c in enumerate(all_cols)
                   if i < cfg['response_col_index'] and c not in exclude]
        else:
            cols = [c for c in df.columns if c not in exclude]
    return df[cols].values.astype(float), cols


def peel_trajectory(X, D, S, rng, alpha_peel, min_support):
    P = X.shape[1]
    idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx) * (1 - alpha_peel) < min_support:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, alpha_peel), np.quantile(xp, 1 - alpha_peel)
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


def build_boxes(X, D, cfg):
    rng = np.random.default_rng(cfg['seed_prim'])
    boxes, total, done = [], len(cfg['s_options']) * cfg['t_per_size'], 0
    for S in cfg['s_options']:
        for _ in range(cfg['t_per_size']):
            idx, obj = peel_trajectory(X, D, S, rng, cfg['alpha_peel'], cfg['min_support'])
            if len(idx) >= cfg['min_support']:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 300 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


def build_linkage(boxes, N):
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    if N > 5000:
        Mem = np.zeros((M, N), dtype=np.int32)
        for i, b in enumerate(boxes):
            Mem[i, b['idx']] = 1
        inter = (Mem @ Mem.T).astype(float)
    else:
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


def eval_ecc_rho_dbar(eff, M):
    if not eff:
        return None
    rho = sum(c['n'] for c in eff) / M
    ecc = np.mean([c['gamma'] for c in eff])
    largest = largest_of(eff)
    return dict(ecc=ecc, rho=rho, dbar=largest['dbar'], largest_mem=largest['mem'])


def cluster_range(boxes, X_train, mem):
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    return X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)


def eval_range_on_confirm(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


def eval_all_clusters_on_confirm(boxes, X_train, X_confirm, D_confirm, eff):
    """
    K*의 효과적 클러스터 '전체'를 confirmation으로 채점.
    각 클러스터의 범위를 confirmation에 각각 적용해 D_conf_k를 구하고,
    training 멤버수(n_k)로 가중평균한다 (원논문의 '여러 클러스터 중 엔지니어가
    재량으로 선택'하는 실제 사용 맥락을 더 정확히 반영하기 위함).
    """
    total_w, weighted_sum = 0.0, 0.0
    for c in eff:
        lo, hi = cluster_range(boxes, X_train, c['mem'])
        d_conf, n_conf = eval_range_on_confirm(lo, hi, X_confirm, D_confirm)
        if not np.isnan(d_conf):
            weighted_sum += c['n'] * d_conf
            total_w += c['n']
    d_agg = weighted_sum / total_w if total_w > 0 else float('nan')
    return d_agg


def full_search_original(raw, M, k_grid, n_memb_options, rho_limit):
    best = None
    for n_memb in n_memb_options:
        for K in k_grid:
            eff = effective_clusters(raw[K], n_memb)
            v = eval_ecc_rho_dbar(eff, M)
            if v is None or v['rho'] < rho_limit:
                continue
            if best is None or v['ecc'] > best[2]['ecc']:
                best = (n_memb, K, v, eff)
    return best


def is_dominated(cand, others):
    ce, cr, cd = cand['ecc'], cand['rho'], cand['dbar']
    for o in others:
        oe, orr, od = o['ecc'], o['rho'], o['dbar']
        if oe >= ce and orr >= cr and od >= cd and (oe > ce or orr > cr or od > cd):
            return True
    return False


def pareto_search(raw, M, k_grid, n_memb_options):
    per_nmemb_best = []
    for n_memb in n_memb_options:
        candidates = []
        for K in k_grid:
            eff = effective_clusters(raw[K], n_memb)
            v = eval_ecc_rho_dbar(eff, M)
            if v is not None:
                candidates.append((K, v, eff))
        if not candidates:
            continue
        vals = [v for _, v, _ in candidates]
        pareto = [(K, v, eff) for K, v, eff in candidates if not is_dominated(v, vals)]
        if not pareto:
            continue
        K_best, v_best, eff_best = max(pareto, key=lambda kve: kve[1]['dbar'])
        per_nmemb_best.append((n_memb, K_best, v_best, eff_best, len(pareto), len(candidates)))
    if not per_nmemb_best:
        return None
    return max(per_nmemb_best, key=lambda r: r[2]['dbar'])


def main():
    print(f'[설정] 데이터셋 = {DATASET_NAME}\n')
    df = pd.read_csv(CSV_PATH)
    X_all, input_cols = get_input_matrix(df, CFG)
    n = len(df)
    print(f'데이터 {n}행, 입력 {len(input_cols)}개\n')

    rng_split = np.random.default_rng(CFG['split_seed'])
    perm = rng_split.permutation(n)
    n_train = int(n * CFG['train_ratio'])
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, X_confirm = X_all[train_idx], X_all[confirm_idx]
    print(f'[0] 분할: training {len(train_idx)}개 / confirmation {len(confirm_idx)}개')

    D_train, fixed = build_desirability(df, CFG, idx=train_idx)
    D_confirm = desirability_with_fixed_ranges(df, CFG, confirm_idx, fixed)

    print('\n[1] training 데이터로 MRS-PRIM 박스 생성')
    boxes = build_boxes(X_train, D_train, CFG)
    print(f'  박스 {len(boxes)}개\n')

    Z, sets, M = build_linkage(boxes, len(train_idx))
    k_grid = list(range(max(int(M * 0.05), 5), int(M * 0.50) + 1, CFG['k_grid_step']))
    raw = precompute_raw_clusters(Z, sets, boxes, M, k_grid)
    print(f'  박스 M={M}, K 격자 {k_grid[0]}~{k_grid[-1]}\n')

    print('=' * 90)
    print(f'  [{DATASET_NAME}] 파레토 최적 vs 원논문 (최종판: N_Memb 넓은격자 + 전체클러스터 채점)')
    print('=' * 90)

    res_o = full_search_original(raw, M, k_grid, CFG['n_memb_options'], CFG['rho_limit'])
    n_memb_o, K_o, v_o, eff_o = res_o
    lo_o, hi_o = cluster_range(boxes, X_train, v_o['largest_mem'])
    d_o_largest, n_o_largest = eval_range_on_confirm(lo_o, hi_o, X_confirm, D_confirm)
    d_o_all = eval_all_clusters_on_confirm(boxes, X_train, X_confirm, D_confirm, eff_o)
    print(f'\n  [기존(rho>={CFG["rho_limit"]})] N_Memb*={n_memb_o}, K*={K_o}, '
          f'효과적클러스터 {len(eff_o)}개')
    print(f'    (a) 최대 클러스터만 : D_conf={d_o_largest:.4f} (n={n_o_largest})')
    print(f'    (b) 전체 가중평균   : D_conf={d_o_all:.4f}')
    if n_memb_o == CFG['n_memb_options'][0]:
        print(f'    [!] N_Memb*가 격자 최솟값({CFG["n_memb_options"][0]})으로 쏠림 -- 경계 문제 의심')

    res_p = pareto_search(raw, M, k_grid, CFG['n_memb_options'])
    n_memb_p, K_p, v_p, eff_p, n_pareto, n_total = res_p
    lo_p, hi_p = cluster_range(boxes, X_train, v_p['largest_mem'])
    d_p_largest, n_p_largest = eval_range_on_confirm(lo_p, hi_p, X_confirm, D_confirm)
    d_p_all = eval_all_clusters_on_confirm(boxes, X_train, X_confirm, D_confirm, eff_p)
    print(f'\n  [파레토최적] N_Memb*={n_memb_p}, K*={K_p}, 효과적클러스터 {len(eff_p)}개 '
          f'(파레토 {n_pareto}/{n_total}개)')
    print(f'    (a) 최대 클러스터만 : D_conf={d_p_largest:.4f} (n={n_p_largest})')
    print(f'    (b) 전체 가중평균   : D_conf={d_p_all:.4f}')
    if n_memb_p == CFG['n_memb_options'][0]:
        print(f'    [!] N_Memb*가 격자 최솟값({CFG["n_memb_options"][0]})으로 쏠림 -- 경계 문제 의심')

    print('\n' + '=' * 90)
    print('  결론 (채점 방식별 비교)')
    print('=' * 90)
    print(f'  (a) 최대 클러스터만 기준 : 차이(파레토-기존) = {d_p_largest-d_o_largest:+.4f}')
    print(f'  (b) 전체 가중평균 기준   : 차이(파레토-기존) = {d_p_all-d_o_all:+.4f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
