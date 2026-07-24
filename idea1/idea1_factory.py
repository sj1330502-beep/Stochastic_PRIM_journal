"""
================================================================================
아이디어 1번 최종 통합 파이프라인 — Multi-stage Factory Process 데이터
================================================================================
콘크리트 버전(idea1_master.py)과 동일한 흐름을 factory 데이터(14088행,
입력 41개, 반응 15개 다중반응)에 적용한다.

콘크리트 버전 대비 변경점
  · desirability: 반응 1개(NTB) -> 반응 15개를 원논문 식(1) 기하평균으로 집계
    (각 반응의 Target = 데이터에 실재하는 Setpoint 값)
  · subspace 크기: 원논문 비율(전체 변수의 50~75%)을 41변수에 맞게 조정
  · min_support: 데이터 규모(14088)에 비례해 250으로 설정
  · 유사도 계산은 행렬곱으로 벡터화 (박스 수 x 관측치 수가 커서 필요)

흐름
  [0] 데이터 분할: training(70%) / confirmation(30%)
  [1] training 만으로 MRS-PRIM 박스 생성
  [2] N_Memb x K 완전 격자탐색으로 단일 최적 (N_Memb*, K*) 선정
  [3] 그 지점의 효과적 클러스터에 수축 알고리즘(1~3단계) 적용
  [4] confirmation 실측 desirability로 Hold-out 검증
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor

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

# --- 데이터 분할 ---
TRAIN_RATIO = 0.7
SPLIT_SEED = 0

# --- 박스 생성 (원논문 비율을 41변수/14088행에 맞춤) ---
ALPHA_PEEL = 0.05
MIN_SUPPORT = 250
S_OPTIONS = (22, 27, 33)
T_PER_SIZE = 250
SEED_PRIM = 1

# --- 클러스터링 완전탐색 ---
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20

# --- 수축 알고리즘 [2~3단계] ---
MC_SAMPLES, ALPHA_Q = 10000, 5
DELTA, BAND = 0.05, 0.10
EPS, VMIN_RATIO, MAX_ITER = 0.05, 0.20, 300

TOPN_CLUSTERS = 10


# ============================================= 다중반응 desirability (원논문 식1)
def build_multi_desirability(df, idx=None):
    """
    idx 가 주어지면 그 행들만 사용(예: training 행 인덱스). Target(Setpoint)은
    전체 데이터 기준으로 고정해 training/confirmation에 동일하게 적용한다.
    """
    d_list = []
    targets = []
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
    """이미 산출된 targets(전체 데이터 기준)를 이용해 특정 idx 구간의 D를 계산"""
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


# ============================================= 클러스터링 (벡터화 유사도 + 완전탐색)
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


# ============================================= 수축 알고리즘 [1~3단계]
def eval_stage2(lo, hi, D_orig, surrogate, targets, rng):
    pts = rng.uniform(lo, hi, size=(MC_SAMPLES, len(lo)))
    y_pred = surrogate.predict(pts)   # (MC_SAMPLES, 15) 다중반응 예측
    d_list = []
    for i, (target, lower, upper) in enumerate(targets):
        y = y_pred[:, i]
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
        d_list.append(np.clip(d, 1e-6, 1.0))
    D_mat = np.column_stack(d_list)
    dpred = np.exp(np.mean(np.log(D_mat), axis=1))
    dD = D_orig - np.percentile(dpred, ALPHA_Q)
    return dD, pts, dpred


def shrink_stage3(lo0, hi0, D_orig, surrogate, targets, rng):
    lo, hi = lo0.copy(), hi0.copy()
    P = len(lo)
    dD, pts, dpred = eval_stage2(lo, hi, D_orig, surrogate, targets, rng)
    V0 = np.prod(np.maximum(hi0 - lo0, 1e-12))
    Vmin = V0 * VMIN_RATIO
    for _ in range(MAX_ITER):
        V = np.prod(np.maximum(hi - lo, 1e-12))
        if dD <= EPS or V <= Vmin:
            break
        span = hi - lo
        best_face, best_dens = None, np.inf
        for p in range(P):
            if span[p] <= 1e-9:
                continue
            b = BAND * span[p]
            dens_lo = np.mean(dpred[pts[:, p] <= lo[p] + b])
            dens_hi = np.mean(dpred[pts[:, p] >= hi[p] - b])
            for face, dens in (('lo', dens_lo), ('hi', dens_hi)):
                if dens < best_dens:
                    best_dens, best_face = dens, (p, face)
        if best_face is None:
            break
        p, face = best_face
        step = DELTA * span[p]
        if face == 'lo':
            lo[p] = min(lo[p] + step, hi[p] - 1e-9)
        else:
            hi[p] = max(hi[p] - step, lo[p] + 1e-9)
        dD, pts, dpred = eval_stage2(lo, hi, D_orig, surrogate, targets, rng)
    return lo0, hi0, lo, hi


# ============================================= Hold-out
def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else np.nan
    return d_mean, n


# ============================================= 실행
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('continuous_factory_process.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH)
    X_all = df[INPUT_COLS].values.astype(float)
    n = len(df)
    print(f'데이터 {n}행, 입력 {X_all.shape[1]}개, 반응 {N_RESPONSES}개(다중반응)')

    # ---------- [0] 분할 ----------
    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, X_confirm = X_all[train_idx], X_all[confirm_idx]
    print(f'[0] 분할: training {len(train_idx)}개 / confirmation {len(confirm_idx)}개')

    # Target(Setpoint)은 전체 데이터 기준으로 한번만 산출 후 공유
    D_train, targets = build_multi_desirability(df, idx=train_idx)
    D_confirm = desirability_from_targets(df, confirm_idx, targets)
    print(f'  training desirability 평균={D_train.mean():.3f}')

    # ---------- [1] 박스 생성 ----------
    print('\n[1] training 데이터로 MRS-PRIM 박스 생성')
    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드')
    else:
        print(f'  S∈{S_OPTIONS}, 총 {len(S_OPTIONS)*T_PER_SIZE} trials (수 분 소요)')
        boxes = build_boxes(X_train, D_train)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'  박스 {len(boxes)}개 생성 및 저장')

    print('  대리모델(RandomForest, 다중출력) 학습 중 ...')
    Y_train_multi = np.column_stack([
        df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)[train_idx]
        for i in range(N_RESPONSES)
    ])
    surrogate = RandomForestRegressor(n_estimators=100, random_state=SEED_PRIM,
                                      n_jobs=-1).fit(X_train, Y_train_multi)
    print(f'  대리모델 R^2(평균, training)={surrogate.score(X_train, Y_train_multi):.3f}')

    # ---------- [2] N_Memb x K 완전탐색 ----------
    print('\n[2] N_Memb x K 완전 격자탐색')
    Z, sets, M = build_linkage(boxes, len(df))
    all_candidates, final = full_grid_search(Z, sets, M)

    print(f'\n  {"N_Memb":>7} {"K*":>6} {"N_eff":>6} {"ECC":>8} {"rho_eff":>8}')
    for n_memb, K_star, N_eff, ecc, rho in all_candidates:
        mark = '  <-- 전체 최적' if (n_memb, K_star) == (final[0], final[1]) else ''
        print(f'  {n_memb:>7} {K_star:>6} {N_eff:>6} {ecc:>8.4f} {rho:>8.3f}{mark}')

    n_memb_star, K_star, N_eff_star, ecc_star, rho_star = final
    print(f'\n  ▶ 최종 채택: N_Memb*={n_memb_star}, K*={K_star} '
          f'(N_eff={N_eff_star}, ECC={ecc_star:.4f}, rho_eff={rho_star:.3f})')

    # ---------- [3],[4] 수축 + Hold-out ----------
    print(f'\n[3~4] (N_Memb*={n_memb_star}, K*={K_star}) 효과적 클러스터에 '
          f'수축 알고리즘 적용 + Hold-out 검증')
    rng = np.random.default_rng()
    lab = fcluster(Z, K_star, criterion='maxclust')
    clusters = [(c, np.where(lab == c)[0]) for c in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= n_memb_star]
    clusters.sort(key=lambda x: -len(x[1]))
    clusters = clusters[:TOPN_CLUSTERS]

    print(f'\n  {"클러스터":>7} {"멤버":>4} {"D_conf(전)":>11} {"D_conf(후)":>11} '
          f'{"개선폭":>8} {"n_conf(전)":>10} {"n_conf(후)":>10}')
    diffs, improved = [], 0
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        D_orig = np.mean([boxes[i]['dbar'] for i in mem])

        lo0_, hi0_, lo, hi = shrink_stage3(lo0, hi0, D_orig, surrogate, targets, rng)
        d_before, n_before = holdout_eval(lo0_, hi0_, X_confirm, D_confirm)
        d_after, n_after = holdout_eval(lo, hi, X_confirm, D_confirm)

        if not (np.isnan(d_before) or np.isnan(d_after)):
            diff = d_after - d_before
            diffs.append(diff)
            if diff > 0:
                improved += 1
        else:
            diff = float('nan')
        print(f'  {cid:>7} {len(mem):>4} {d_before:>11.4f} {d_after:>11.4f} '
              f'{diff:>+8.4f} {n_before:>10} {n_after:>10}')

    n_tested = len(diffs)
    if n_tested:
        print(f'\n  → 개선된 클러스터: {improved}/{n_tested} '
              f'({100*improved/n_tested:.1f}%), 평균 개선폭={np.mean(diffs):+.4f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
