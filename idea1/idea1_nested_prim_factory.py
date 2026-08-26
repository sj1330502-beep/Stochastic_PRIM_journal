"""
================================================================================
아이디어 1번 최종판 — Multi-stage Factory Process 데이터
================================================================================
RATIO2 후보는 outer training 내부의 inner validation에서 선택하고, outer
confirmation은 선택된 RATIO2의 최종 성능을 평가할 때 한 번만 사용한다.

  outer training
    ├─ inner training   : MRS-PRIM 박스/클러스터 생성
    └─ inner validation : RATIO2 선택
  outer confirmation    : 선택 완료 후 최종 성능 평가

완전 실측 기반이며 예측모델/몬테카를로는 사용하지 않는다.
================================================================================
"""
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'continuous_factory_process.csv')
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache',
                         'stochastic_prim_journal', 'idea1')
CACHE_VERSION = 'ratio2-nested-v1'

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
INNER_TRAIN_RATIO = 0.7
INNER_SPLIT_SEED = 1

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


# ============================================= 데이터 분할
def split_indices(indices, train_ratio, seed):
    indices = np.asarray(indices, dtype=int)
    perm = np.random.default_rng(seed).permutation(len(indices))
    n_train = int(len(indices) * train_ratio)
    if n_train == 0 or n_train == len(indices):
        raise ValueError('training과 hold-out에 각각 한 개 이상의 관측치가 필요합니다.')
    return indices[perm[:n_train]], indices[perm[n_train:]]


# ============================================= 다중반응 desirability (원논문 식1)
def build_multi_desirability(df, idx):
    """학습 인덱스에서만 desirability 경계와 target을 추정한다."""
    idx = np.asarray(idx, dtype=int)
    d_list, targets = [], []
    for i in range(N_RESPONSES):
        y_all = df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)
        y = y_all[idx]
        sp = df[f'Stage1.Output.Measurement{i}.U.Setpoint'].iloc[idx]
        nonzero = sp[sp != 0]
        target = float(nonzero.mode().iloc[0]) if len(nonzero) else float(sp.mean())
        lower, upper = float(y.min()), float(y.max())
        if upper <= target or target <= lower:
            target = float(np.median(y))
        targets.append((target, lower, upper))
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


def boxes_cache_path(label, X, D):
    """분할 데이터와 PRIM 설정이 같을 때만 같은 캐시를 사용한다."""
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode())
    digest.update(repr((ALPHA_PEEL, MIN_SUPPORT, S_OPTIONS,
                        T_PER_SIZE, SEED_PRIM)).encode())
    for array in (X, D):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return os.path.join(CACHE_DIR, f'factory_{label}_{digest.hexdigest()[:16]}.npy')


def load_or_build_boxes(label, X, D):
    cache_path = boxes_cache_path(label, X, D)
    if os.path.exists(cache_path):
        boxes = list(np.load(cache_path, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드: {os.path.basename(cache_path)}')
        return boxes

    boxes = build_boxes(X, D)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, np.array(boxes, dtype=object), allow_pickle=True)
    print(f'  박스 {len(boxes)}개 생성 및 저장: {os.path.basename(cache_path)}')
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


def find_effective_clusters(boxes, n_observations):
    Z, sets, M = build_linkage(boxes, n_observations)
    all_candidates, final = full_grid_search(Z, sets, M)
    n_memb_star, K_star, N_eff_star, ecc_star, rho_star = final
    lab = fcluster(Z, K_star, criterion='maxclust')
    clusters = [(c, np.where(lab == c)[0]) for c in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= n_memb_star]
    clusters.sort(key=lambda x: -len(x[1]))
    return clusters[:TOPN_CLUSTERS], final


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
def holdout_eval(lo, hi, X_holdout, D_holdout):
    mask = np.all((X_holdout >= lo) & (X_holdout <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_holdout[mask].mean()) if n > 0 else np.nan
    return d_mean, n


# ============================================= RATIO2 스윕 (inner validation 전용)
def sweep_ratio2(clusters, boxes, X_train, D_train, X_validation, D_validation):
    print('=' * 78)
    print('  RATIO2 스윕 (inner validation에서 선택)')
    print('=' * 78)
    print(f'  {"RATIO2":>7} {"평균D_valid(후)":>15} {"평균개선폭":>11} '
          f'{"평균n_valid(후)":>15} {"n_valid=0 비율":>15}')

    precomputed = []
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
        X_sub, D_sub = X_train[in_box], D_train[in_box]
        n_box = int(in_box.sum())
        d_before, _ = holdout_eval(lo0, hi0, X_validation, D_validation)
        precomputed.append((cid, lo0, hi0, X_sub, D_sub, n_box, d_before))

    sweep_results = []
    for ratio2 in RATIO2_GRID:
        d_afters, diffs, n_afters, zero_count = [], [], [], 0
        for cid, lo0, hi0, X_sub, D_sub, n_box, d_before in precomputed:
            min_support2 = max(MIN_SUPPORT2_FLOOR, int(n_box * ratio2))
            lo, hi, _, _ = peel_within_box(X_sub, D_sub, lo0, hi0, min_support2)
            d_after, n_after = holdout_eval(lo, hi, X_validation, D_validation)
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
              f'{mean_n:>15.1f} {zero_ratio:>14.1%}')

    valid = [r for r in sweep_results if r[4] < 0.5 and not np.isnan(r[1])]
    best = max(valid, key=lambda r: r[1]) if valid else max(
        sweep_results, key=lambda r: (not np.isnan(r[1]), r[1]))
    print(f'\n  ▶ 채택 RATIO2 = {best[0]} '
          f'(평균 D_valid(후)={best[1]:.4f}, 평균 개선폭={best[2]:+.4f}, '
          f'평균 n_valid(후)={best[3]:.1f})')
    return best[0]


def evaluate_on_confirmation(clusters, boxes, X_train, D_train,
                             X_confirm, D_confirm, ratio2):
    print(f'\n[4] RATIO2={ratio2} 고정 후 outer confirmation 최종 평가')
    print(f'\n  {"클러스터":>7} {"멤버":>4} {"n_box":>6} {"min_sup2":>8} '
          f'{"D_conf(전)":>11} {"D_conf(후)":>11} {"개선폭":>8} '
          f'{"n_conf(전)":>10} {"n_conf(후)":>10}')
    diffs, improved = [], 0
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
        X_sub, D_sub = X_train[in_box], D_train[in_box]
        n_box = int(in_box.sum())
        min_support2 = max(MIN_SUPPORT2_FLOOR, int(n_box * ratio2))

        lo, hi, _, _ = peel_within_box(X_sub, D_sub, lo0, hi0, min_support2)
        d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)
        d_after, n_after = holdout_eval(lo, hi, X_confirm, D_confirm)

        diff = d_after - d_before if not (np.isnan(d_before) or np.isnan(d_after)) else float('nan')
        if not np.isnan(diff):
            diffs.append(diff)
            if diff > 0:
                improved += 1
        print(f'  {cid:>7} {len(mem):>4} {n_box:>6} {min_support2:>8} '
              f'{d_before:>11.4f} {d_after:>11.4f} {diff:>+8.4f} '
              f'{n_before:>10} {n_after:>10}')

    if diffs:
        print(f'\n  → 개선된 클러스터: {improved}/{len(diffs)} '
              f'({100 * improved / len(diffs):.1f}%), 평균 개선폭={np.mean(diffs):+.4f}')


# ============================================= 실행
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('continuous_factory_process.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH)
    X_all = df[INPUT_COLS].values.astype(float)
    n = len(df)
    print(f'데이터 {n}행, 입력 {X_all.shape[1]}개, 반응 {N_RESPONSES}개(다중반응)')

    outer_train_idx, confirm_idx = split_indices(
        np.arange(n), TRAIN_RATIO, SPLIT_SEED)
    inner_train_idx, validation_idx = split_indices(
        outer_train_idx, INNER_TRAIN_RATIO, INNER_SPLIT_SEED)
    print(f'[0] outer 분할: training {len(outer_train_idx)}개 / '
          f'confirmation {len(confirm_idx)}개')
    print(f'    inner 분할: training {len(inner_train_idx)}개 / '
          f'validation {len(validation_idx)}개')

    # ---------- inner validation에서 RATIO2 선택 ----------
    X_inner_train = X_all[inner_train_idx]
    D_inner_train, inner_targets = build_multi_desirability(df, inner_train_idx)
    X_validation = X_all[validation_idx]
    D_validation = desirability_from_targets(df, validation_idx, inner_targets)

    print('\n[1] inner training 데이터로 RATIO2 선택용 MRS-PRIM 박스 생성')
    inner_boxes = load_or_build_boxes('inner', X_inner_train, D_inner_train)
    inner_clusters, inner_final = find_effective_clusters(
        inner_boxes, len(X_inner_train))
    print(f'  ▶ inner 채택: N_Memb*={inner_final[0]}, K*={inner_final[1]} '
          f'(N_eff={inner_final[2]}, ECC={inner_final[3]:.4f}, '
          f'rho_eff={inner_final[4]:.3f})')

    print('\n[2] outer confirmation을 보지 않고 RATIO2 선택')
    best_ratio2 = sweep_ratio2(
        inner_clusters, inner_boxes, X_inner_train, D_inner_train,
        X_validation, D_validation)

    # ---------- outer training 전체로 최종 파이프라인 재구축 ----------
    X_outer_train = X_all[outer_train_idx]
    D_outer_train, outer_targets = build_multi_desirability(df, outer_train_idx)
    X_confirm = X_all[confirm_idx]
    D_confirm = desirability_from_targets(df, confirm_idx, outer_targets)

    print('\n[3] outer training 전체로 최종 MRS-PRIM 재구축')
    outer_boxes = load_or_build_boxes('outer', X_outer_train, D_outer_train)
    outer_clusters, outer_final = find_effective_clusters(
        outer_boxes, len(X_outer_train))
    print(f'  ▶ outer 채택: N_Memb*={outer_final[0]}, K*={outer_final[1]} '
          f'(N_eff={outer_final[2]}, ECC={outer_final[3]:.4f}, '
          f'rho_eff={outer_final[4]:.3f})')

    evaluate_on_confirmation(
        outer_clusters, outer_boxes, X_outer_train, D_outer_train,
        X_confirm, D_confirm, best_ratio2)

    print('\n완료. (RATIO2 선택=inner validation, 최종 평가=outer confirmation)')


if __name__ == '__main__':
    main()
