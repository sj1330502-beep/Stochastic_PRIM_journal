"""
================================================================================
아이디어 1번 최종 통합 파이프라인 v3 — 가상포인트 중첩밀도 기반 수축 (원문 정확 해석)
================================================================================
idea1_master.py 와 전체 구조(데이터분할/N_Memb x K 완전탐색/Hold-out)는 동일.
[3단계] '어디를 깎을지' 결정하는 기준을 원문 그대로 재구현했다:

  원문: "경계면 주변에 위치한 가상 포인트들의 중첩 밀도를 확인한다"
  -> 주어는 '가상 포인트', 그 포인트들이 '무엇과' 중첩되는지가 핵심이며,
     이는 원래 멤버 subregion(박스)들이다.

  v1(이전): 가상 포인트의 '대리모델 예측 desirability'가 낮은 쪽을 깎음
            -> 대리모델의 회귀/외삽 불확실성에 전적으로 의존.
  v2(중간, 폐기): 멤버 박스 구간이 경계 밴드와 1차원씩 겹치는지만 확인
            -> 변수별 marginal 검사라 다차원 구조를 놓치는 문제.
  v3(본 파일): 경계 밴드에 가상 포인트를 뿌리고, 그 각 포인트가 원래
            멤버 박스 중 하나에 '다차원 전부 동시에' 속하는지(중첩)를
            판정해 그 비율(밀도)을 구한다. 대리모델 예측이 필요 없고
            (단순 좌표 소속 판정), 다차원 구조도 올바르게 반영한다.

역할 분담
  [3-1,3-2] 어디를 깎을지        -> 가상포인트가 멤버 박스에 중첩되는 밀도
  [3-3,3-4] 재평가 및 종료(ΔD,V) -> 몬테카를로 + 대리모델 (원안 2단계 방식)
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
BOX_CACHE = os.path.join(HERE, 'boxes_master_train.npy')   # v1과 동일 캐시 재사용 가능

TRAIN_RATIO = 0.7
SPLIT_SEED = 0

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 25

MC_SAMPLES, ALPHA_Q = 10000, 5
DELTA, BAND = 0.05, 0.10          # BAND: 경계면 근처 '밀도 측정 밴드' 폭 (구간폭의 10%)
EPS, VMIN_RATIO, MAX_ITER = 0.05, 0.20, 300

TOPN_CLUSTERS = 10


# ============================================= desirability (Koo Table1 NTB)
def make_desirability(y_all):
    lower, upper = float(y_all.min()), float(y_all.max())
    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= 60.0)
        d[m1] = (y[m1] - lower) / (60.0 - lower)
        m2 = (y > 60.0) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - 60.0)
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
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 300 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 클러스터링 + N_Memb x K 완전탐색
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
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average'), sets, M


def full_grid_search(Z, sets, M):
    all_candidates = []
    for n_memb in N_MEMB_OPTIONS:
        best_for_this = None
        for K in range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP):
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


# ============================================= [2단계] 몬테카를로+대리모델로 ΔD, V 평가
def eval_stage2(lo, hi, D_orig, surrogate, d_func, rng):
    pts = rng.uniform(lo, hi, size=(MC_SAMPLES, len(lo)))
    dpred = d_func(surrogate.predict(pts))
    dD = D_orig - np.percentile(dpred, ALPHA_Q)
    V = np.prod(np.maximum(hi - lo, 1e-12))
    return dD, V


# ============================================= [3단계] 가상포인트-중첩밀도 기반 수축 (v3, 원문 정확 해석)
def virtual_overlap_density(lo, hi, member_los, member_his, p, face, rng, n_probe=2000):
    """
    경계면(p, face) 근처 밴드에 가상 포인트를 뿌리고, 그 포인트들이 원래
    멤버 subregion들(다차원 박스) 중 하나에라도 실제로 속하는(중첩되는)
    비율을 계산한다. -> '가상 포인트들의 중첩 밀도'(원문 그대로)
    """
    span = hi - lo
    if span[p] <= 1e-9:
        return np.inf
    band = BAND * span[p]
    lo_b, hi_b = lo.copy(), hi.copy()
    if face == 'lo':
        hi_b[p] = lo[p] + band
    else:
        lo_b[p] = hi[p] - band

    pts = rng.uniform(lo_b, hi_b, size=(n_probe, len(lo)))
    # 각 가상포인트가 '어떤 멤버 박스 하나에라도(다차원 전부) 속하는지' 판정
    covered = np.zeros(n_probe, dtype=bool)
    for m_lo, m_hi in zip(member_los, member_his):
        covered |= np.all((pts >= m_lo) & (pts <= m_hi), axis=1)
    return covered.mean()   # 중첩 밀도 (0~1)


def shrink_stage3(lo0, hi0, D_orig, member_los, member_his,
                  surrogate, d_func, rng):
    lo, hi = lo0.copy(), hi0.copy()
    P = len(lo)
    dD, V = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)
    V0 = np.prod(np.maximum(hi0 - lo0, 1e-12))
    Vmin = V0 * VMIN_RATIO

    for _ in range(MAX_ITER):
        if dD <= EPS or V <= Vmin:
            break

        # --- [3-1,3-2] 경계 탐색: 가상포인트 중첩 밀도가 가장 낮은 면 선택 ---
        best_face, best_density = None, np.inf
        for p in range(P):
            for face in ('lo', 'hi'):
                dens = virtual_overlap_density(lo, hi, member_los, member_his,
                                               p, face, rng)
                if dens < best_density:
                    best_density, best_face = dens, (p, face)
        if best_face is None or not np.isfinite(best_density):
            break

        # --- 최적 방향 수축: 그 면을 delta% 안쪽으로 슬라이딩 ---
        p, face = best_face
        step = DELTA * (hi[p] - lo[p])
        if face == 'lo':
            lo[p] = min(lo[p] + step, hi[p] - 1e-9)
        else:
            hi[p] = max(hi[p] - step, lo[p] + 1e-9)

        # --- [3-3,3-4] 선호도 재평가 및 종료조건(ΔD, V)은 몬테카를로+대리모델 ---
        dD, V = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)

    return lo0, hi0, lo, hi


# ============================================= Hold-out
def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else np.nan
    return d_mean, n


# ============================================= 실행
def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X_all = df.iloc[:, :8].values.astype(float)
    y_all = df.iloc[:, 8].values.astype(float)
    feat_names = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
                  'CoarseAgg', 'FineAgg', 'Age']

    rng_split = np.random.default_rng(SPLIT_SEED)
    n = len(y_all)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_confirm, y_confirm = X_all[confirm_idx], y_all[confirm_idx]
    print(f'[0] 데이터 분할: training {len(y_train)}개 / confirmation {len(y_confirm)}개')

    d_func, lower, upper = make_desirability(y_all)
    D_train = d_func(y_train)
    D_confirm = d_func(y_confirm)

    print('\n[1] training 데이터로 MRS-PRIM 박스 생성')
    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드')
    else:
        boxes = build_boxes(X_train, D_train)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'  박스 {len(boxes)}개 생성 및 저장')

    surrogate = RandomForestRegressor(n_estimators=100, random_state=SEED_PRIM
                                      ).fit(X_train, y_train)
    print(f'  대리모델(training만 학습) R^2={surrogate.score(X_train, y_train):.3f}')

    print('\n[2] N_Memb x K 완전 격자탐색 (원논문 식 12)')
    Z, sets, M = build_linkage(boxes)
    all_candidates, final = full_grid_search(Z, sets, M)

    print(f'\n  {"N_Memb":>7} {"K*":>6} {"N_eff":>6} {"ECC":>8} {"rho_eff":>8}')
    for n_memb, K_star, N_eff, ecc, rho in all_candidates:
        mark = '  <-- 전체 최적' if (n_memb, K_star) == (final[0], final[1]) else ''
        print(f'  {n_memb:>7} {K_star:>6} {N_eff:>6} {ecc:>8.4f} {rho:>8.3f}{mark}')

    n_memb_star, K_star, N_eff_star, ecc_star, rho_star = final
    print(f'\n  ▶ 최종 채택: N_Memb*={n_memb_star}, K*={K_star} '
          f'(N_eff={N_eff_star}, ECC={ecc_star:.4f}, rho_eff={rho_star:.3f})')

    print(f'\n[3~4] (N_Memb*={n_memb_star}, K*={K_star}) 효과적 클러스터에 '
          f'중첩밀도 기반 수축 알고리즘(v3) 적용 + Hold-out 검증')
    rng = np.random.default_rng()
    lab = fcluster(Z, K_star, criterion='maxclust')
    clusters = [(c, np.where(lab == c)[0]) for c in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= n_memb_star]
    clusters.sort(key=lambda x: -len(x[1]))
    clusters = clusters[:TOPN_CLUSTERS]

    print(f'\n  {"클러스터":>7} {"멤버":>4} {"D_conf(전)":>11} {"D_conf(후)":>11} '
          f'{"개선폭":>8} {"n_conf(전)":>10} {"n_conf(후)":>10}')
    diffs, improved = [], 0
    first_result = None
    for cid, mem in clusters:
        # 멤버 박스들의 개별 경계 (중첩 밀도 계산에 사용)
        member_los = np.array([X_train[boxes[i]['idx']].min(axis=0) for i in mem])
        member_his = np.array([X_train[boxes[i]['idx']].max(axis=0) for i in mem])

        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        D_orig = np.mean([boxes[i]['dbar'] for i in mem])

        lo0_, hi0_, lo, hi = shrink_stage3(lo0, hi0, D_orig, member_los, member_his,
                                           surrogate, d_func, rng)
        d_before, n_before = holdout_eval(lo0_, hi0_, X_confirm, D_confirm)
        d_after, n_after = holdout_eval(lo, hi, X_confirm, D_confirm)

        if first_result is None:
            first_result = (cid, lo0_, hi0_, lo, hi)

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
    print(f'\n  → 개선된 클러스터: {improved}/{n_tested} '
          f'({100*improved/n_tested:.1f}%), 평균 개선폭={np.mean(diffs):+.4f}')

    cid, lo0_, hi0_, lo, hi = first_result
    print(f'\n  [클러스터 {cid}] 단일 대표 운전구간 (통합 전 → 수축 후, v3)')
    print(f'  {"변수":>10} {"통합(전)":>20} {"수축(후)":>20}')
    for p, nm in enumerate(feat_names):
        print(f'  {nm:>10} [{lo0_[p]:>8.1f},{hi0_[p]:>8.1f}]   '
              f'[{lo[p]:>8.1f},{hi[p]:>8.1f}]')

    print('\n완료.')


if __name__ == '__main__':
    main()
