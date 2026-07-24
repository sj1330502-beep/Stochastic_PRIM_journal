"""
================================================================================
아이디어 2번: Tversky vs Jaccard 비교 — Multi-stage Factory Process 데이터
================================================================================
데이터: continuous_factory_process.csv (Kaggle, 14088행)
  입력  : Stage1 이전 공정변수 44개
          (AmbientConditions 2 + Machine1/2/3 공정변수 39 + FirstStage
           CombinerOperation 온도 3)
  반응  : Stage1.Output.Measurement0~14 (Actual), 총 15개 — 각 반응마다
          Setpoint(목표값)가 데이터에 실재하므로, 이를 NTB desirability의
          Target으로 그대로 사용 (원 논문 Table1 NTB 공식과 완전히 일치).
  집계  : 원논문 식(1) 가중기하평균으로 15개 desirability를 D_i 하나로 종합
          (다중반응 MRO 구조 — 원논문과 동일한 성격)

콘크리트 버전 대비 변경점
  · 다중반응 desirability 집계 추가 (콘크리트는 반응 1개라 없었음)
  · Target을 임의값이 아니라 데이터의 실제 Setpoint로 사용
  · subspace 크기를 44변수에 맞게 조정 (원논문 비율 50~75% 유지)
  · 유사도 계산을 행렬곱으로 벡터화 (M~1500, N~14088로 커서 이중 for문은
    비현실적으로 느림 → boolean 멤버십 행렬의 내적으로 교집합을 한번에 계산)
  · 데이터가 커서 min_support 등 비율은 원논문과 동일하게 유지하되 절대값만
    데이터 크기에 비례 조정

Tversky 대칭화는 콘크리트 버전과 동일한 방식(작은/큰 박스 판정, W_SMALL/W_LARGE).
================================================================================
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'continuous_factory_process.csv')

# --- 입력/반응 컬럼 정의 ---
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
N_RESPONSES = 15   # Stage1.Output.Measurement0~14

# --- 박스 생성 (원논문 MRS-PRIM 구조) ---
ALPHA_PEEL = 0.05
MIN_SUPPORT = 250          # N=14088 기준, 원논문 min_support 비율(~1.8%)과 유사
S_OPTIONS = (22, 27, 33)   # P=44의 50~75% (원논문 비율 유지)
T_PER_SIZE = 250           # 총 750 trials (데이터가 커서 대형 T*K는 비현실적)
SEED_PRIM = 1

# --- 클러스터링 (원논문) ---
N_MEMB, RHO_LIMIT = 5, 0.6

# --- Tversky 대칭 가중치 ---
W_SMALL, W_LARGE = 0.8, 0.2


# ============================================= 다중반응 desirability (원논문 식1)
def build_multi_desirability(df):
    """
    각 반응 Measurement{i} : NTB, Target=데이터의 실제 Setpoint(최빈 비0값),
    lower/upper = 그 반응의 실제 관측 min/max.
    D_i = 15개 개별 desirability의 (동일가중) 기하평균  [원논문 식(1)]
    """
    d_list = []
    for i in range(N_RESPONSES):
        y = df[f'Stage1.Output.Measurement{i}.U.Actual'].values.astype(float)
        sp = df[f'Stage1.Output.Measurement{i}.U.Setpoint']
        nonzero = sp[sp != 0]
        target = float(nonzero.mode().iloc[0]) if len(nonzero) else float(sp.mean())
        lower, upper = float(y.min()), float(y.max())
        if upper <= target or target <= lower:
            # target이 범위 경계와 겹치면 desirability가 항상 0이 되는 걸 방지
            lower, upper = y.min(), y.max()
            target = float(np.median(y))
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= target)
        d[m1] = (y[m1] - lower) / (target - lower) if target > lower else 1.0
        m2 = (y > target) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - target) if upper > target else 1.0
        d_list.append(np.clip(d, 1e-6, 1.0))   # 기하평균 위해 0 방지
    D_mat = np.column_stack(d_list)             # (N, 15)
    D_agg = np.exp(np.mean(np.log(D_mat), axis=1))  # 동일가중 기하평균
    return D_agg


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
    boxes, total = [], len(S_OPTIONS) * T_PER_SIZE
    done = 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 100 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 벡터화된 유사도 계산 (대용량용)
def membership_matrix(boxes, N):
    """boxes -> (M, N) boolean 멤버십 행렬 (M개 박스가 어떤 관측치를 포함하는지)"""
    M = len(boxes)
    Mem = np.zeros((M, N), dtype=np.int32)
    for i, b in enumerate(boxes):
        Mem[i, b['idx']] = 1
    return Mem


def similarity_matrix_fast(boxes, N, method):
    """행렬곱으로 교집합을 한번에 계산 (이중 for문 대비 대폭 빠름)"""
    Mem = membership_matrix(boxes, N)               # (M, N)
    inter = Mem @ Mem.T                              # (M, M) 교집합 크기
    sup = np.array([b['support'] for b in boxes], dtype=float)
    inter = inter.astype(float)

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


# ============================================= 원논문 ECC 평가
def evaluate(Z, boxes, inter_mat, K):
    sup = np.array([b['support'] for b in boxes], dtype=float)
    M = len(boxes)
    lab = fcluster(Z, K, criterion='maxclust')
    gams, Meff, neff = [], 0, 0
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        n = len(mem)
        if n < N_MEMB:
            continue
        # 합집합 근사: 포함-배제 대신, 실제 union은 인덱스로 정확 계산
        U = set()
        for m in mem:
            U.update(boxes[m]['idx'].tolist())
        I = set(boxes[mem[0]]['idx'].tolist())
        for m in mem[1:]:
            I &= set(boxes[m]['idx'].tolist())
        gams.append(len(I) / len(U) if U else 0.0)
        Meff += n; neff += 1
    return dict(ECC=float(np.mean(gams)) if gams else 0.0,
               N_eff=neff, rho=Meff / M)


def best_by_ecc(Z, boxes, inter_mat):
    M = len(boxes)
    best = None
    for K in range(max(int(M * 0.05), 5), int(M * 0.50) + 1, 20):
        r = evaluate(Z, boxes, inter_mat, K)
        if r['rho'] >= RHO_LIMIT and (best is None or r['ECC'] > best[1]['ECC']):
            best = (K, r)
    return best


# ============================================= 크기 비대칭 진단
def diagnose_asymmetry(boxes, inter_mat):
    sup = np.array([b['support'] for b in boxes], dtype=float)
    M = len(boxes)
    iu = np.triu_indices(M, k=1)
    inter = inter_mat[iu]
    mask = inter > 0
    small = np.minimum(sup[iu[0]], sup[iu[1]])[mask]
    large = np.maximum(sup[iu[0]], sup[iu[1]])[mask]
    inter_ov = inter[mask]
    ratios = large / small
    contain = np.sum((inter_ov >= 0.8 * small) & (ratios >= 2.0))
    return dict(sup_min=int(sup.min()), sup_max=int(sup.max()),
                sup_ratio=sup.max() / sup.min(),
                pair_ratio_med=float(np.median(ratios)) if len(ratios) else 1.0,
                pair_ratio_max=float(ratios.max()) if len(ratios) else 1.0,
                contain_pairs=int(contain), overlap_pairs=int(mask.sum()))


# ============================================= 실행
def main():
    t0 = time.time()
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('continuous_factory_process.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH)
    X = df[INPUT_COLS].values.astype(float)
    N = len(df)
    print(f'데이터 {N}행, 입력 {X.shape[1]}개, 반응 {N_RESPONSES}개(Stage1)')

    D = build_multi_desirability(df)
    print(f'다중반응 desirability(기하평균): 평균={D.mean():.3f}, '
          f'표준편차={D.std():.3f}')

    print('\nMRS-PRIM 박스 생성 (S in', S_OPTIONS, ', 총',
          len(S_OPTIONS) * T_PER_SIZE, 'trials) ...')
    boxes = build_boxes(X, D)
    print(f'  박스 {len(boxes)}개  (경과 {time.time()-t0:.0f}s)')

    print('\n유사도 행렬 계산 (벡터화) ...')
    sim_j, inter_j = similarity_matrix_fast(boxes, N, 'jaccard')
    sim_t, inter_t = similarity_matrix_fast(boxes, N, 'tversky')
    print(f'  완료 (경과 {time.time()-t0:.0f}s)')

    print('\n' + '=' * 66)
    print('  [사전진단] 크기 비대칭 쌍이 존재하는가?')
    print('=' * 66)
    dg = diagnose_asymmetry(boxes, inter_j)
    print(f'  support 범위           : {dg["sup_min"]} ~ {dg["sup_max"]} '
          f'(최대/최소 = {dg["sup_ratio"]:.2f}배)')
    print(f'  겹치는 쌍 크기비(중앙) : {dg["pair_ratio_med"]:.2f}배, '
          f'최대 {dg["pair_ratio_max"]:.2f}배')
    print(f'  비대칭 포함 쌍         : {dg["contain_pairs"]}개 / '
          f'겹치는 쌍 {dg["overlap_pairs"]}개 '
          f'({100*dg["contain_pairs"]/max(dg["overlap_pairs"],1):.2f}%)')
    if dg['contain_pairs'] == 0:
        print('  ▶ 비대칭 포함 쌍 없음 → Tversky 효과 기대 어려움')
    else:
        print('  ▶ 비대칭 포함 쌍 존재 → Tversky가 다르게 처리할 여지 있음')

    print('\n' + '=' * 66)
    print('  [비교] 원논문 Jaccard  vs  아이디어2 Tversky')
    print('=' * 66)
    results = {}
    for name, sim, inter in (('jaccard', sim_j, inter_j), ('tversky', sim_t, inter_t)):
        Z = make_linkage(sim)
        best = best_by_ecc(Z, boxes, inter)
        results[name] = best
        tag = '원논문' if name == 'jaccard' else '아이디어2'
        if best is None:
            print(f'\n  [{name.upper()}] ({tag}) — rho_Limit 만족하는 K 없음')
            continue
        K_star, r = best
        print(f'\n  [{name.upper()}] ({tag})')
        print(f'    K*={K_star}  ECC={r["ECC"]:.4f}  '
              f'N_eff={r["N_eff"]}  rho_eff={r["rho"]:.3f}')

    if results['jaccard'] and results['tversky']:
        je, te = results['jaccard'][1]['ECC'], results['tversky'][1]['ECC']
        print('\n' + '=' * 66 + '\n  [판정]\n' + '=' * 66)
        print(f'  ECC : Jaccard {je:.4f} vs Tversky {te:.4f} (차이 {te-je:+.4f})')
        if abs(te - je) < 1e-3:
            print('  ▶ 사실상 동일 (크기 비대칭 미미)')
        elif te > je:
            print('  ▶ Tversky 우세 → 아이디어2가 이 데이터에서 유리')
        else:
            print('  ▶ Jaccard 우세 → 이 데이터에선 Tversky가 불리')

    print(f'\n총 소요시간 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
