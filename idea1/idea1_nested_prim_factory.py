"""
================================================================================
아이디어 1번 최종판 — Multi-stage Factory Process 데이터
================================================================================
콘크리트에서 확정한 최종 설계(완전 실측 기반, 예측모델/몬테카를로 없음,
RATIO2 기반 2차 PRIM, deterministic peeling, N_Memb x K 완전탐색)를
factory 데이터(14088행, 입력 41개, 다중반응 15개)에 그대로 이식한다.

콘크리트 버전 대비 변경점
  · desirability: 반응 15개를 원논문 식(1) 기하평균으로 집계
    (각 반응 Target = 데이터의 실제 Setpoint)
  · 박스 생성 하이퍼파라미터: min_support=250, subspace=(22,27,33) 등
    데이터 규모/변수 수에 비례 조정 (기존 factory 실험과 동일 설정)
  · 유사도 계산은 행렬곱으로 벡터화 (M, N 이 커서 필요)
  · RATIO2(2차 PRIM의 min_support 비율)는 콘크리트에서 스윕한 0.4를
    기본값으로 사용하되, 데이터가 다르므로 스윕도 함께 수행해 재확인한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'continuous_factory_process.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_factory_train.npy')

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

N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20

MIN_SUPPORT2_FLOOR = 20
RATIO2_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]   # 콘크리트에서 채택한 0.4 포함해 재스윕

TOPN_CLUSTERS = 10


# ============================================= 다중반응 desirability (원논문 식1)
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


# ============================================= MRS-PRIM 박스 생성 (1차, stochastic)
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


# ============================================= 클러스터링 (벡터화 유사도) + 완전탐색
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


def full_grid_search(Z, sets, M):
    all_candidates = []
    for n_memb in N_MEMB_OPTIONS:
        best_for_this = None
        for K in range(max(int(M * 0.05), 5), int(M * 0.50) + 1, K_GRID_STEP):
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
        raise RuntimeError('rho_Limit 을 만족하는 (N_Memb, K) 조합이 없습니다.')
    final = max(all_candidates, key=lambda c: c[3])
    return all_candidates, final


# ============================================= [2단계] 통합박스 내 2차 PRIM (완전 실측, deterministic)
def peel_within_box(X_sub, D_sub, lo0, hi0, min_support2):
    P = X_sub.shape[1]
    idx = np.arange(len(D_sub))
    lo, hi = lo0.copy(), hi0.copy()
    best_idx, best_lo, best_hi, best_obj = idx.copy(), lo.copy(), hi.copy(), D_sub.mean()
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < min_support2:
            break
        cand_keep, cand_obj, cand_p, cand_bound = None, -np.inf, None, None
        for p in range(P):
            xp = X_sub[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
            for keep, bound in ((idx[xp > lo_q], ('lo', lo_q)),
                               (idx[xp < hi_q], ('hi', hi_q))):
                if min_support2 <= len(keep) < len(idx):
                    obj = D_sub[keep].mean()
                    if obj > cand_obj:
                        cand_obj, cand_keep, cand_p, cand_bound = obj, keep, p, bound
        if cand_keep is None:
            break
        idx = cand_keep
        side, val = cand_bound
        if side == 'lo':
            lo[cand_p] = val
        else:
            hi[cand_p] = val
        if cand_obj > best_obj:
            best_obj = cand_obj
            best_idx, best_lo, best_hi = idx.copy(), lo.copy(), hi.copy()
    return best_lo, best_hi, best_obj, len(best_idx)


# ============================================= Hold-out
def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else np.nan
    return d_mean, n


# ============================================= RATIO2 스윕
def sweep_ratio2(clusters, boxes, X_train, D_train, X_confirm, D_confirm):
    print('=' * 78)
    print('  RATIO2 스윕 (factory 데이터에서 재확인)')
    print('=' * 78)
    print(f'  {"RATIO2":>7} {"평균D_conf(후)":>15} {"평균개선폭":>11} '
          f'{"평균n_conf(후)":>15} {"n_conf=0 비율":>13}')

    precomputed = []
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
        X_sub, D_sub = X_train[in_box], D_train[in_box]
        n_box = int(in_box.sum())
        d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)
        precomputed.append((cid, lo0, hi0, X_sub, D_sub, n_box, d_before))

    sweep_results = []
    for ratio2 in RATIO2_GRID:
        d_afters, diffs, n_afters, zero_count = [], [], [], 0
        for cid, lo0, hi0, X_sub, D_sub, n_box, d_before in precomputed:
            min_support2 = max(MIN_SUPPORT2_FLOOR, int(n_box * ratio2))
            lo, hi, obj_final, n_final = peel_within_box(X_sub, D_sub, lo0, hi0, min_support2)
            d_after, n_after = holdout_eval(lo, hi, X_confirm, D_confirm)
            if n_after == 0:
                zero_count += 1
                continue
            d_afters.append(d_after)
            n_afters.append(n_after)
            if not np.isnan(d_before):
                diffs.append(d_after - d_before)
        mean_d = np.mean(d_afters) if d_afters else float('nan')
        mean_diff = np.mean(diffs) if diffs else float('nan')
        mean_n = np.mean(n_afters) if n_afters else float('nan')
        zero_ratio = zero_count / len(precomputed)
        sweep_results.append((ratio2, mean_d, mean_diff, mean_n, zero_ratio))
        print(f'  {ratio2:>7.1f} {mean_d:>15.4f} {mean_diff:>+11.4f} '
              f'{mean_n:>15.1f} {zero_ratio:>12.1%}')

    valid = [r for r in sweep_results if r[4] < 0.5 and not np.isnan(r[1])]
    best = max(valid, key=lambda r: r[1]) if valid else max(
        sweep_results, key=lambda r: (not np.isnan(r[1]), r[1]))
    print(f'\n  ▶ 채택 RATIO2 = {best[0]} '
          f'(평균 D_conf(후)={best[1]:.4f}, 평균 개선폭={best[2]:+.4f}, '
          f'평균 n_conf(후)={best[3]:.1f})')
    return best[0]


# ============================================= 실행
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('continuous_factory_process.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH)
    X_all = df[INPUT_COLS].values.astype(float)
    n = len(df)
    print(f'데이터 {n}행, 입력 {X_all.shape[1]}개, 반응 {N_RESPONSES}개(다중반응)')

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
    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드')
    else:
        boxes = build_boxes(X_train, D_train)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'  박스 {len(boxes)}개 생성 및 저장')

    print('\n[2] N_Memb x K 완전 격자탐색')
    Z, sets, M = build_linkage(boxes, len(df))
    all_candidates, final = full_grid_search(Z, sets, M)
    n_memb_star, K_star, N_eff_star, ecc_star, rho_star = final
    print(f'  ▶ 채택: N_Memb*={n_memb_star}, K*={K_star} '
          f'(N_eff={N_eff_star}, ECC={ecc_star:.4f}, rho_eff={rho_star:.3f})')

    lab = fcluster(Z, K_star, criterion='maxclust')
    clusters = [(c, np.where(lab == c)[0]) for c in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= n_memb_star]
    clusters.sort(key=lambda x: -len(x[1]))
    clusters = clusters[:TOPN_CLUSTERS]

    best_ratio2 = sweep_ratio2(clusters, boxes, X_train, D_train, X_confirm, D_confirm)

    print(f'\n[3] 채택된 RATIO2={best_ratio2} 로 최종 수축 수행')
    print(f'\n  {"클러스터":>7} {"멤버":>4} {"n_box":>6} {"min_sup2":>8} {"D_conf(전)":>11} '
          f'{"D_conf(후)":>11} {"개선폭":>8} {"n_conf(전)":>10} {"n_conf(후)":>10}')
    diffs, improved = [], 0
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
        X_sub, D_sub = X_train[in_box], D_train[in_box]
        n_box = int(in_box.sum())
        min_support2 = max(MIN_SUPPORT2_FLOOR, int(n_box * best_ratio2))

        lo, hi, obj_final, n_final = peel_within_box(X_sub, D_sub, lo0, hi0, min_support2)
        d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)
        d_after, n_after = holdout_eval(lo, hi, X_confirm, D_confirm)

        diff = d_after - d_before if not (np.isnan(d_before) or np.isnan(d_after)) else float('nan')
        if not np.isnan(diff):
            diffs.append(diff)
            if diff > 0:
                improved += 1
        print(f'  {cid:>7} {len(mem):>4} {n_box:>6} {min_support2:>8} {d_before:>11.4f} '
              f'{d_after:>11.4f} {diff:>+8.4f} {n_before:>10} {n_after:>10}')

    if diffs:
        print(f'\n  → 개선된 클러스터: {improved}/{len(diffs)} '
              f'({100*improved/len(diffs):.1f}%), 평균 개선폭={np.mean(diffs):+.4f}')

    print('\n완료. (예측모델/몬테카를로 미사용, 전 과정 실측 기반)')


if __name__ == '__main__':
    main()
