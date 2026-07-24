"""
================================================================================
IEEM 원논문 방법론 + 아이디어1 정밀 레시피 수축 알고리즘 [1~3단계 전체]
================================================================================
파이프라인
  원논문 재현 : MRS-PRIM 박스 생성 → 경험적 Jaccard → average-linkage → K*(ECC)
  아이디어1   :
    [1단계] 통합 경계 : 클러스터 멤버 박스를 감싸는 최소 외접 박스 B_global
    [2단계] 품질 리스크 평가 : 몬테카를로 가상포인트 → 대리모델 예측 →
            ΔD = D̄_orig - Quantile_α(D_pred),  하이퍼볼륨 V
    [3단계] 밀도 기반 정밀 수축 :
        1. 경계 탐색   : 각 변수 외곽 경계면 주변 가상포인트 중첩 밀도 확인
        2. 최적방향수축: 밀도가 0에 가까운 경계를 δ% 안쪽으로 슬라이딩
        3. 선호도 재평가: 살아남은 가상포인트로 ΔD, V 재계산
        4. 종료        : ΔD ≤ ε  또는  V ≤ V_min  (아이디어안 문서 그대로)

설정 메모
  · desirability : Koo Table1 표준 NTB, Target=60, lower/upper=데이터 실제 min/max
  · 몬테카를로   : 매 실행 랜덤 (시드 고정 안 함)
  · 수축 대상    : 내부 파편화가 존재하는 성긴 granularity 에서 시연
                   (K* 는 내부가 응집적이라 수축 여지가 작음 — 결과에 함께 표시)
--------------------------------------------------------------------------------
사용법
  python -m pip install numpy pandas scipy scikit-learn
  concrete_dataset.csv 를 같은 폴더에 두고:  python idea1_full.py
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

# --- 박스 생성 (원논문 MRS-PRIM) ---
ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

# --- 클러스터링 (원논문) ---
N_MEMB, RHO_LIMIT = 5, 0.6

# --- desirability NTB ---
NTB_TARGET, NTB_SHAPE = 60.0, 1.0

# --- 아이디어1 [2~3단계] 하이퍼파라미터 ---
MC_SAMPLES  = 10000    # 몬테카를로 가상 포인트 수
ALPHA_Q     = 5        # ΔD 하위 분위수 (%)
DELTA       = 0.05     # [3단계] 경계 수축 스텝 δ (구간폭의 5%)
BAND        = 0.10     # 경계면 근처 밴드 폭 (구간폭의 10%)
EPS         = 0.05     # 종료조건: ΔD ≤ ε
VMIN_RATIO  = 0.20     # 종료조건: V ≤ V0 * VMIN_RATIO
MAX_ITER    = 300      # 수축 반복 상한


# ============================================= desirability (Koo Table1 NTB)
def make_desirability(y_all):
    lower, upper = float(y_all.min()), float(y_all.max())
    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= NTB_TARGET)
        d[m1] = ((y[m1] - lower) / (NTB_TARGET - lower)) ** NTB_SHAPE
        m2 = (y > NTB_TARGET) & (y <= upper)
        d[m2] = ((upper - y[m2]) / (upper - NTB_TARGET)) ** NTB_SHAPE
        return np.clip(d, 0.0, 1.0)
    return d_func, lower, upper


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
    boxes = []
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append(dict(idx=idx, support=len(idx), dbar=obj))
    return boxes


# ============================================= 클러스터링 + K*
def build_linkage(boxes):
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)
    sim = inter / (sup[:, None] + sup[None, :] - inter)
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average'), sets, M


def find_kstar(Z, sets, M):
    best = None
    for K in range(int(M * 0.05), int(M * 0.50) + 1, 25):
        lab = fcluster(Z, K, criterion='maxclust')
        gams, Meff, neff = [], 0, 0
        for k in np.unique(lab):
            mem = np.where(lab == k)[0]
            if len(mem) < N_MEMB:
                continue
            I = set.intersection(*[sets[m] for m in mem])
            U = set.union(*[sets[m] for m in mem])
            gams.append(len(I)/len(U) if U else 0); Meff += len(mem); neff += 1
        if neff and Meff / M >= RHO_LIMIT:
            ecc = np.mean(gams)
            if best is None or ecc > best[1]:
                best = (K, ecc)
    return best[0]


# ============================================= 아이디어1 [1~3단계]
def hypervolume_log(lo, hi):
    """로그 하이퍼볼륨 (변수 스케일 차가 커 언더플로 방지)"""
    return float(np.sum(np.log(np.maximum(hi - lo, 1e-12))))


def eval_stage2(lo, hi, D_orig, surrogate, d_func, rng):
    """[2단계] 몬테카를로로 ΔD, V, (가상포인트, 내부마스크) 반환"""
    pts = rng.uniform(lo, hi, size=(MC_SAMPLES, len(lo)))
    dpred = d_func(surrogate.predict(pts))
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)   # 전부 True (직육면체)
    dpred_in = dpred[inside]
    if len(dpred_in) == 0:
        return np.inf, hypervolume_log(lo, hi), pts, dpred, inside
    dD = D_orig - np.percentile(dpred_in, ALPHA_Q)
    return dD, hypervolume_log(lo, hi), pts, dpred, inside


def shrink_stage3(lo0, hi0, D_orig, surrogate, d_func, rng):
    """[3단계] 밀도 기반 정밀 수축. 아이디어안 종료조건: ΔD ≤ ε 또는 V ≤ V_min."""
    lo, hi = lo0.copy(), hi0.copy()
    P = len(lo)
    dD, V, pts, dpred, inside = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)
    logV0 = V
    logVmin = logV0 + np.log(VMIN_RATIO)
    hist = [(dD, 1.0)]

    for _ in range(MAX_ITER):
        # 종료조건 (아이디어안 문서): ΔD ≤ ε 또는 V ≤ V_min
        if dD <= EPS or V <= logVmin:
            break
        span = hi - lo
        # 1. 경계 탐색: 각 변수 상/하 경계면 밴드 내 가상포인트 밀도
        best_face, best_dens = None, np.inf
        for p in range(P):
            if span[p] <= 1e-9:
                continue
            b = BAND * span[p]
            dens_lo = np.mean(dpred[(pts[:, p] <= lo[p] + b)])   # 하한면 근처 평균 desirability 밀도
            dens_hi = np.mean(dpred[(pts[:, p] >= hi[p] - b)])   # 상한면 근처
            for face, dens in (('lo', dens_lo), ('hi', dens_hi)):
                if dens < best_dens:
                    best_dens, best_face = dens, (p, face)
        if best_face is None:
            break
        # 2. 최적 방향 수축: 밀도 최저 경계를 δ% 안쪽으로
        p, face = best_face
        step = DELTA * span[p]
        if face == 'lo':
            lo[p] = min(lo[p] + step, hi[p] - 1e-9)
        else:
            hi[p] = max(hi[p] - step, lo[p] + 1e-9)
        # 3. 선호도 재평가
        dD, V, pts, dpred, inside = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)
        hist.append((dD, np.exp(V - logV0)))

    return dict(lo0=lo0, hi0=hi0, lo=lo, hi=hi,
                dD=dD, logV0=logV0, logV=V, Vratio=np.exp(V - logV0),
                n_iter=len(hist) - 1, hist=hist)


def run_idea1(lab, boxes, X, d_func, surrogate, rng, label, feat_names, topN=8):
    print(f'\n{"="*70}\n[{label}]  1~3단계 정밀 레시피 수축\n{"="*70}')
    print(f'  {"클러스터":>7} {"멤버":>4} {"D_orig":>7} '
          f'{"ΔD(전)":>7} {"ΔD(후)":>7} {"V후/V0":>7} {"수축":>5}')
    clusters = [(cid, np.where(lab == cid)[0]) for cid in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= N_MEMB]
    clusters.sort(key=lambda x: -len(x[1]))

    saved = []
    for cid, mem in clusters[:topN]:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X[all_idx].min(axis=0), X[all_idx].max(axis=0)
        D_orig = np.mean([boxes[i]['dbar'] for i in mem])
        dD0, _, _, _, _ = eval_stage2(lo0, hi0, D_orig, surrogate, d_func, rng)
        r = shrink_stage3(lo0, hi0, D_orig, surrogate, d_func, rng)
        saved.append((cid, mem, r))
        print(f'  {cid:>7} {len(mem):>4} {D_orig:>7.3f} '
              f'{dD0:>7.3f} {r["dD"]:>7.3f} {r["Vratio"]:>7.3f} {r["n_iter"]:>5}')

    # 첫 클러스터 대표구간 상세
    cid, mem, r = saved[0]
    print(f'\n  [클러스터 {cid}] 단일 대표 운전구간  (통합 전 → 수축 후)')
    print(f'  {"변수":>10} {"통합(전)":>20} {"수축(후)":>20} {"폭축소":>7}')
    for p, nm in enumerate(feat_names):
        w0 = r["hi0"][p] - r["lo0"][p]
        w1 = r["hi"][p] - r["lo"][p]
        shrink = (1 - w1 / w0) * 100 if w0 > 0 else 0
        print(f'  {nm:>10} [{r["lo0"][p]:>7.1f},{r["hi0"][p]:>7.1f}] '
              f'[{r["lo"][p]:>7.1f},{r["hi"][p]:>7.1f}] {shrink:>6.1f}%')
    return saved


# ============================================= 실행
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError('concrete_dataset.csv 를 같은 폴더에 두세요.')
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    feat_names = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
                  'CoarseAgg', 'FineAgg', 'Age']
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)

    d_func, lower, upper = make_desirability(y)
    D = d_func(y)
    print(f'데이터 {len(y)}행 | NTB: Target={NTB_TARGET}, '
          f'lower={lower:.2f}, upper={upper:.2f} | D평균={D.mean():.3f}')

    surrogate = RandomForestRegressor(n_estimators=100,
                                      random_state=SEED_PRIM).fit(X, y)
    print(f'대리모델 R^2={surrogate.score(X, y):.3f}')

    print('\nMRS-PRIM 박스 생성 ...')
    boxes = build_boxes(X, D)
    Z, sets, M = build_linkage(boxes)
    K_star = find_kstar(Z, sets, M)
    print(f'  박스 {M}개, K*(ECC 최대화)={K_star}')

    rng = np.random.default_rng()   # 몬테카를로 매 실행 랜덤

    # 성긴 granularity: 내부 파편화 존재 → 수축 대상이 잘 보임
    K_coarse = max(10, int(M * 0.05))
    lab_coarse = fcluster(Z, K_coarse, criterion='maxclust')
    run_idea1(lab_coarse, boxes, X, d_func, surrogate, rng,
              f'성긴 granularity K={K_coarse}', feat_names)

    # 원논문 K*: 내부 응집적 → 수축 여지 작음 (대비용)
    lab_star = fcluster(Z, K_star, criterion='maxclust')
    run_idea1(lab_star, boxes, X, d_func, surrogate, rng,
              f'원논문 K*={K_star}', feat_names)

    print('\n완료. (몬테카를로 랜덤 → ΔD/수축결과는 실행마다 소폭 변동)')


if __name__ == '__main__':
    main()
