"""
================================================================================
아이디어 2번: Jaccard(원논문) vs Tversky(아이디어2)  ECC 비교
================================================================================
K* 탐색 방식은 그대로(원논문 식12: rho_eff>=rho_Limit 제약 하 ECC 최대화하는
K를 훑어서 찾는 휴리스틱). 이 스크립트는 그 K*에서의 ECC를 두 거리방식으로
공정 비교하는 데만 집중한다.

두 데이터셋 겸용:
  MODE = 'concrete'  → concrete_dataset.csv  (단일반응 NTB, Target=60)
  MODE = 'factory'   → continuous_factory_process.csv
                        (다중반응 15개, Target=Setpoint, 기하평균 집계)

비교 항목: K*, ECC, N_eff, rho_eff  +  사전진단(크기 비대칭 존재 여부)
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================= 실행 모드 선택
MODE = 'concrete'    # 'concrete' 또는 'factory' 로 바꿔서 실행

# --- 공통 클러스터링 설정 (원논문) ---
N_MEMB, RHO_LIMIT = 5, 0.6
ALPHA_PEEL = 0.05
SEED_PRIM = 1

# --- Tversky 대칭 가중치 (작은/큰 박스 판정) ---
W_SMALL, W_LARGE = 0.8, 0.2

# --- 데이터셋별 설정 ---
CONFIG = {
    'concrete': dict(
        csv='concrete_dataset.csv', min_support=100,
        s_options=(4, 5, 6), t_per_size=667,
    ),
    'factory': dict(
        csv='continuous_factory_process.csv', min_support=250,
        s_options=(22, 27, 33), t_per_size=250,
    ),
}


# ============================================= desirability
def desirability_ntb(y, target, lower, upper):
    y = np.asarray(y, dtype=float)
    d = np.zeros_like(y)
    m1 = (y >= lower) & (y <= target)
    d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
    m2 = (y > target) & (y <= upper)
    d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
    return np.clip(d, 0.0, 1.0)


def load_concrete():
    df = pd.read_csv(os.path.join(HERE, CONFIG['concrete']['csv']),
                     encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    D = desirability_ntb(y, target=60.0, lower=y.min(), upper=y.max())
    return X, D


def load_factory():
    input_cols = [
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
    df = pd.read_csv(os.path.join(HERE, CONFIG['factory']['csv']))
    X = df[input_cols].values.astype(float)

    d_list = []
    for i in range(15):
        y = df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)
        sp = df[f'Stage1.Output.Measurement{i}.U.Setpoint']
        nonzero = sp[sp != 0]
        target = float(nonzero.mode().iloc[0]) if len(nonzero) else float(sp.mean())
        lower, upper = float(y.min()), float(y.max())
        if upper <= target or target <= lower:
            lower, upper, target = y.min(), y.max(), float(np.median(y))
        d_list.append(np.clip(desirability_ntb(y, target, lower, upper), 1e-6, 1.0))
    D_mat = np.column_stack(d_list)
    D = np.exp(np.mean(np.log(D_mat), axis=1))     # 원논문 식(1) 기하평균
    return X, D


# ============================================= MRS-PRIM 박스 생성
def peel_trajectory(X, D, S, min_support, rng):
    P = X.shape[1]
    idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < min_support:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
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


def build_boxes(X, D, s_options, t_per_size, min_support):
    rng = np.random.default_rng(SEED_PRIM)
    boxes = []
    total = len(s_options) * t_per_size
    done = 0
    for S in s_options:
        for _ in range(t_per_size):
            idx, obj = peel_trajectory(X, D, S, min_support, rng)
            if len(idx) >= min_support:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 200 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 벡터화된 유사도 (Jaccard/Tversky)
def membership_matrix(boxes, N):
    M = len(boxes)
    Mem = np.zeros((M, N), dtype=np.int32)
    for i, b in enumerate(boxes):
        Mem[i, b['idx']] = 1
    return Mem


def similarity_matrix(boxes, N, method):
    Mem = membership_matrix(boxes, N)
    inter = (Mem @ Mem.T).astype(float)
    sup = np.array([b['support'] for b in boxes], dtype=float)
    a, b = sup[:, None], sup[None, :]
    only_i, only_j = a - inter, b - inter

    if method == 'jaccard':
        denom = inter + only_i + only_j
    elif method == 'tversky':
        small_only = np.minimum(only_i, only_j)
        large_only = np.maximum(only_i, only_j)
        denom = inter + W_SMALL * small_only + W_LARGE * large_only
    else:
        raise ValueError(method)

    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom != 0)
    sim = (sim + sim.T) / 2
    return sim, inter


def make_linkage(sim):
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average')


# ============================================= 원논문 ECC (식 9~11)
def evaluate(Z, boxes, K):
    M = len(boxes)
    lab = fcluster(Z, K, criterion='maxclust')
    gams, Meff, neff = [], 0, 0
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        if len(mem) < N_MEMB:
            continue
        U = set()
        for m in mem:
            U.update(boxes[m]['idx'].tolist())
        I = set(boxes[mem[0]]['idx'].tolist())
        for m in mem[1:]:
            I &= set(boxes[m]['idx'].tolist())
        gams.append(len(I) / len(U) if U else 0.0)
        Meff += len(mem); neff += 1
    return dict(ECC=float(np.mean(gams)) if gams else 0.0,
               N_eff=neff, rho=Meff / M)


def find_kstar(Z, boxes):
    """원논문 식(12): 휴리스틱 격자탐색으로 K* 찾기 (기존 방식 그대로 유지)"""
    M = len(boxes)
    best = None
    for K in range(max(int(M * 0.05), 5), int(M * 0.50) + 1, 20):
        r = evaluate(Z, boxes, K)
        if r['rho'] >= RHO_LIMIT and (best is None or r['ECC'] > best[1]['ECC']):
            best = (K, r)
    return best


# ============================================= 사전진단 (참고용)
def diagnose_asymmetry(boxes, inter_mat):
    sup = np.array([b['support'] for b in boxes], dtype=float)
    M = len(boxes)
    iu = np.triu_indices(M, k=1)
    inter = inter_mat[iu]
    mask = inter > 0
    if mask.sum() == 0:
        return dict(sup_ratio=1.0, contain_pairs=0, overlap_pairs=0)
    small = np.minimum(sup[iu[0]], sup[iu[1]])[mask]
    large = np.maximum(sup[iu[0]], sup[iu[1]])[mask]
    inter_ov = inter[mask]
    contain = np.sum((inter_ov >= 0.8 * small) & (large / small >= 2.0))
    return dict(sup_ratio=sup.max() / sup.min(),
                contain_pairs=int(contain), overlap_pairs=int(mask.sum()))


# ============================================= 실행
def main():
    cfg = CONFIG[MODE]
    print(f'[모드: {MODE}]  데이터 로드 ...')
    X, D = load_concrete() if MODE == 'concrete' else load_factory()
    N = len(D)
    print(f'  N={N}, 입력변수={X.shape[1]}개, D 평균={D.mean():.3f}')

    print('\nMRS-PRIM 박스 생성 ...')
    boxes = build_boxes(X, D, cfg['s_options'], cfg['t_per_size'], cfg['min_support'])
    print(f'  박스 {len(boxes)}개')

    sim_j, inter_j = similarity_matrix(boxes, N, 'jaccard')
    sim_t, inter_t = similarity_matrix(boxes, N, 'tversky')

    dg = diagnose_asymmetry(boxes, inter_j)
    print(f'\n[사전진단] support 최대/최소={dg["sup_ratio"]:.2f}배, '
          f'비대칭 포함쌍={dg["contain_pairs"]}/{dg["overlap_pairs"]}')

    print('\n' + '=' * 60)
    print('  ECC 비교 (K*는 원래 하던 휴리스틱 탐색 그대로)')
    print('=' * 60)
    for name, sim in (('JACCARD (원논문)', sim_j), ('TVERSKY (아이디어2)', sim_t)):
        Z = make_linkage(sim)
        best = find_kstar(Z, boxes)
        if best is None:
            print(f'  {name:<20}: rho_Limit 만족 K 없음')
            continue
        K_star, r = best
        print(f'  {name:<20}: K*={K_star:<5} ECC={r["ECC"]:.4f}  '
              f'N_eff={r["N_eff"]:<4} rho_eff={r["rho"]:.3f}')

    print('\n완료.')


if __name__ == '__main__':
    main()
