"""
================================================================================
아이디어 2번: Tversky vs Jaccard(원논문) 거리측정 방식 비교
================================================================================
목적
  원논문은 박스 간 거리를 '경험적 Jaccard'로 잰다.
  아이디어2는 크기 비대칭에 강건한 'Tversky 지수'를 제안한다.
  → 동일한 MRS-PRIM 박스에 두 거리만 바꿔 적용하고, 어느 쪽이
    더 나은 클러스터(높은 ECC, 살아나는 소형영역)를 만드는지 비교한다.

공정 비교를 위한 3가지 장치
  (1) 앞단(박스 생성·linkage 절차·ECC 정의)은 두 방식에 100% 동일.
      오직 '유사도 공식'만 교체.
  (2) ECC 는 원논문 정의(효과적 클러스터 Γ의 평균) 그대로 사용.
  (3) 비교 전에 '크기 비대칭 쌍이 실제로 존재하는지' 먼저 진단.
      → 비대칭이 없으면 Tversky 효과가 원리적으로 나타나지 않으므로,
        그 사실 자체를 정직하게 리포트한다.

Tversky 대칭화
  각 쌍에서 '작은 박스 / 큰 박스'를 크기로 판정하여,
  큰 박스에만 있는 부분(A_large_only)에 낮은 가중치를 준다.
  → 순서 무관 + 대칭 보장 (계층 클러스터링 입력 조건 충족).

사용법
  python -m pip install numpy pandas scipy
  concrete_dataset.csv 를 같은 폴더에 두고:  python idea2_compare.py
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

# --- 박스 생성 (원논문 MRS-PRIM) ---
ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

# --- 클러스터링 (원논문) ---
N_MEMB, RHO_LIMIT = 5, 0.6

# --- desirability NTB ---
NTB_TARGET, NTB_SHAPE = 60.0, 1.0

# --- Tversky 대칭 가중치: 큰 박스에만 있는 부분에 낮은 가중치 ---
#     w_small : 작은 박스에만 있는 부분(강하게 벌점)
#     w_large : 큰 박스에만 있는 부분(약하게 벌점 → 포함관계 보존)
W_SMALL, W_LARGE = 0.8, 0.2


# ============================================= desirability
def make_desirability(y_all):
    lower, upper = float(y_all.min()), float(y_all.max())
    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= NTB_TARGET)
        d[m1] = ((y[m1]-lower)/(NTB_TARGET-lower))**NTB_SHAPE
        m2 = (y > NTB_TARGET) & (y <= upper)
        d[m2] = ((upper-y[m2])/(upper-NTB_TARGET))**NTB_SHAPE
        return np.clip(d, 0.0, 1.0)
    return d_func


# ============================================= MRS-PRIM 박스 생성
def peel_trajectory(X, D, S, rng):
    P = X.shape[1]; idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx)*(1-ALPHA_PEEL) < MIN_SUPPORT:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1-ALPHA_PEEL)
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
    boxes = []
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
    return boxes


# ============================================= 유사도 (핵심: 두 방식)
def similarity_matrix(boxes, method):
    """method='jaccard'(원논문) 또는 'tversky'(아이디어2, 대칭화)"""
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)

    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)

    a = sup[:, None]            # 박스 i 크기
    b = sup[None, :]            # 박스 j 크기
    only_i = a - inter          # i 에만 있는 관측치
    only_j = b - inter          # j 에만 있는 관측치

    if method == 'jaccard':
        denom = inter + only_i + only_j                 # = |합집합|
    elif method == 'tversky':
        # 대칭화: 각 쌍에서 작은/큰 박스를 크기로 판정
        small_only = np.minimum(only_i, only_j)         # 작은 박스에만 있는 쪽
        large_only = np.maximum(only_i, only_j)         # 큰 박스에만 있는 쪽
        denom = inter + W_SMALL*small_only + W_LARGE*large_only
    else:
        raise ValueError(method)

    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom != 0)
    sim = (sim + sim.T) / 2
    return sim


def make_linkage(sim):
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average')


# ============================================= 원논문 ECC 로 평가
def evaluate(Z, boxes, K):
    """원논문 정의: 효과적 클러스터(n>=N_MEMB) Γ=|∩|/|∪| 의 평균"""
    sets = [set(b['idx'].tolist()) for b in boxes]
    M = len(boxes)
    lab = fcluster(Z, K, criterion='maxclust')
    gams, Meff, neff, small_kept = [], 0, 0, 0
    for k in np.unique(lab):
        mem = np.where(lab == k)[0]
        if len(mem) < N_MEMB:
            continue
        I = set.intersection(*[sets[m] for m in mem])
        U = set.union(*[sets[m] for m in mem])
        gams.append(len(I)/len(U) if U else 0.0)
        Meff += len(mem); neff += 1
    return dict(ECC=float(np.mean(gams)) if gams else 0.0,
               N_eff=neff, rho=Meff/M)


def best_by_ecc(Z, boxes):
    """원논문 식(12): ρ_eff>=ρ_Limit 제약 하에서 ECC 최대화하는 K* 및 그 지표"""
    M = len(boxes)
    best = None
    for K in range(int(M*0.05), int(M*0.50)+1, 25):
        r = evaluate(Z, boxes, K)
        if r['rho'] >= RHO_LIMIT and (best is None or r['ECC'] > best[1]['ECC']):
            best = (K, r)
    return best


# ============================================= 크기 비대칭 진단
def diagnose_asymmetry(boxes):
    sup = np.array([b['support'] for b in boxes], dtype=float)
    sets = [set(b['idx'].tolist()) for b in boxes]
    M = len(boxes)
    ratios, contain_pairs, total_overlap = [], 0, 0
    for i in range(M):
        for j in range(i+1, M):
            inter = len(sets[i] & sets[j])
            if inter == 0:
                continue
            total_overlap += 1
            small, large = min(sup[i], sup[j]), max(sup[i], sup[j])
            ratios.append(large/small)
            # 작은 박스가 큰 박스에 80% 이상 포함되는데 크기차 2배 이상 → 비대칭 포함
            if inter >= 0.8*small and large/small >= 2.0:
                contain_pairs += 1
    ratios = np.array(ratios) if ratios else np.array([1.0])
    return dict(sup_min=int(sup.min()), sup_max=int(sup.max()),
                sup_ratio=sup.max()/sup.min(),
                pair_ratio_med=float(np.median(ratios)),
                pair_ratio_max=float(ratios.max()),
                contain_pairs=contain_pairs, overlap_pairs=total_overlap)


# ============================================= 실행
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('concrete_dataset.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    D = make_desirability(y)(y)

    print('MRS-PRIM 박스 생성 (원논문 방식) ...')
    boxes = build_boxes(X, D)
    print(f'  박스 {len(boxes)}개')

    # ---------- (3) 크기 비대칭 진단 : 비교의 전제 ----------
    print('\n' + '='*66)
    print('  [사전진단] 크기 비대칭 쌍이 존재하는가?  (Tversky 효과의 전제)')
    print('='*66)
    dg = diagnose_asymmetry(boxes)
    print(f'  박스 support 범위      : {dg["sup_min"]} ~ {dg["sup_max"]} '
          f'(최대/최소 = {dg["sup_ratio"]:.2f}배)')
    print(f'  겹치는 쌍의 크기비(중앙): {dg["pair_ratio_med"]:.2f}배, '
          f'최대 {dg["pair_ratio_max"]:.2f}배')
    print(f'  비대칭 포함 쌍         : {dg["contain_pairs"]}개 '
          f'/ 겹치는 쌍 {dg["overlap_pairs"]}개 '
          f'({100*dg["contain_pairs"]/max(dg["overlap_pairs"],1):.2f}%)')
    if dg['contain_pairs'] == 0:
        print('  ▶ 비대칭 포함 쌍이 없음 → Tversky와 Jaccard 결과가 거의 동일할 것')
    else:
        print('  ▶ 비대칭 포함 쌍 존재 → Tversky가 이를 다르게 처리할 여지 있음')

    # ---------- (1)(2) 동일 앞단에 두 거리만 바꿔 비교 ----------
    print('\n' + '='*66)
    print('  [비교] 원논문 Jaccard  vs  아이디어2 Tversky')
    print('='*66)
    results = {}
    for method in ['jaccard', 'tversky']:
        sim = similarity_matrix(boxes, method)
        Z = make_linkage(sim)
        K_star, r = best_by_ecc(Z, boxes)
        results[method] = (K_star, r)
        tag = '원논문' if method == 'jaccard' else '아이디어2'
        print(f'\n  [{method.upper()}] ({tag})')
        print(f'    K*        = {K_star}')
        print(f'    ECC       = {r["ECC"]:.4f}   (높을수록 조밀)')
        print(f'    N_eff     = {r["N_eff"]}')
        print(f'    rho_eff   = {r["rho"]:.3f}')

    # ---------- 판정 ----------
    je, te = results['jaccard'][1]['ECC'], results['tversky'][1]['ECC']
    print('\n' + '='*66)
    print('  [판정]')
    print('='*66)
    print(f'  ECC : Jaccard {je:.4f}  vs  Tversky {te:.4f}  '
          f'(차이 {te-je:+.4f})')
    if abs(te - je) < 1e-3:
        print('  ▶ 사실상 동일. 이 데이터에는 크기 비대칭이 거의 없어')
        print('    Tversky의 이점이 나타나지 않는다 (사전진단과 일치).')
    elif te > je:
        print('  ▶ Tversky가 더 조밀한 클러스터를 형성 → 아이디어2가 유리')
    else:
        print('  ▶ Jaccard가 더 우수 → 이 데이터에선 Tversky가 불리')
    print('\n  주의: ECC 단독 비교는 참고치. 최종 판단은 hold-out 상의')
    print('        일반화 성능·소형영역 보존까지 함께 봐야 한다.')


if __name__ == '__main__':
    main()
