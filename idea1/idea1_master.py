"""
================================================================================
아이디어 1번 최종 통합 파이프라인
================================================================================
흐름
  [0] 데이터 분할: training(70%) / confirmation(30%)
  [1] training 만으로 MRS-PRIM 박스 생성 (원논문 방식)
  [2] N_Memb x K 완전 격자탐색으로 '전체를 통틀어 가장 좋은' 단일 조합
      (N_Memb*, K*) 을 선정
        - 원논문 식(12): 각 N_Memb에 대해 rho_eff>=rho_Limit 제약 하
          ECC 를 최대화하는 K 를 찾는다.
        - 이렇게 N_Memb 별로 얻어진 (K, ECC) 후보들 중, ECC 가 가장 높은
          단 하나의 (N_Memb*, K*) 를 최종 채택한다.
        - 즉 이전 버전들처럼 N_Memb=5 로 임의 고정하지 않고, 원논문이
          실제로 탐색하는 전체 범위(N_Memb 5~10) x K 격자를 다 훑어서
          '진짜 최적'을 스스로 결정한다.
  [3] 그 (N_Memb*, K*) 에서 나온 효과적 클러스터들에 대해 아이디어1의
      정밀 레시피 수축 알고리즘(1~3단계, 몬테카를로+대리모델)을 적용한다.
  [4] 수축 전/후 대표구간을 confirmation 데이터에 적용해, 실측
      desirability 로 Hold-out 검증을 수행한다 (대리모델과 무관한 검증).

이 구조는 클러스터 선정 단계(N_Memb, K)의 자의성을 없애고, 그 위에서
수축 알고리즘의 효과를 검증하므로 이전 버전들보다 신뢰성이 높다.
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
BOX_CACHE = os.path.join(HERE, 'boxes_master_train.npy')

# --- 데이터 분할 ---
TRAIN_RATIO = 0.7
SPLIT_SEED = 0

# --- 박스 생성 (원논문 MRS-PRIM) ---
ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

# --- 클러스터링 완전탐색 (원논문 Step 4 범위) ---
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 25

# --- 수축 알고리즘 [2~3단계] ---
MC_SAMPLES, ALPHA_Q = 10000, 5
DELTA, BAND = 0.05, 0.10
EPS, VMIN_RATIO, MAX_ITER = 0.05, 0.20, 300

TOPN_CLUSTERS = 10   # 최종 선정된 (N_Memb*, K*) 에서 검토할 클러스터 수(멤버 많은 순)


# ============================================= desirability (NTB, Koo Table1)
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


# ============================================= 클러스터링 (linkage 1회, N_Memb x K 완전탐색)
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
    """
    원논문 식(12)를 N_Memb x K 전체에 대해 수행하고,
    그 중 ECC 가 가장 높은 단일 (N_Memb*, K*) 를 최종 채택한다.
    반환: 전체 후보 테이블, 최종 선택 (n_memb, K, N_eff, ECC, rho_eff)
    """
    all_candidates = []
    for n_memb in N_MEMB_OPTIONS:
        best_for_this_nmemb = None
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
                if best_for_this_nmemb is None or ecc > best_for_this_nmemb[2]:
                    best_for_this_nmemb = (K, neff, ecc, Meff / M)
        if best_for_this_nmemb is not None:
            K_star, N_eff, ecc, rho = best_for_this_nmemb
            all_candidates.append((n_memb, K_star, N_eff, ecc, rho))

    if not all_candidates:
        raise RuntimeError('rho_Limit 을 만족하는 (N_Memb, K) 조합이 없습니다.')

    # 전체 후보 중 ECC 최댓값 = 최종 (N_Memb*, K*)
    final = max(all_candidates, key=lambda c: c[3])
    return all_candidates, final


# ============================================= 수축 알고리즘 [1~3단계]
def eval_stage2(lo, hi, D_orig, surrogate, d_func, rng):
    pts = rng.uniform(lo, hi, size=(MC_SAMPLES, len(lo)))
    dpred = d_func(surrogate.predict(pts))
    dD = D_orig - np.percentile(dpred, ALPHA_Q)
    return dD, pts, dpred


def shrink_stage3(lo0, hi0, D_orig, surrogate, d_func, rng):
    lo, hi = lo0.copy(), hi0.copy()
    P = len(lo)
    dD, pts, dpred = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)
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
        dD, pts, dpred = eval_stage2(lo, hi, D_orig, surrogate, d_func, rng)
    return lo0, hi0, lo, hi


# ============================================= Hold-out 검증
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

    # ---------- [0] 데이터 분할 ----------
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

    # ---------- [1] 박스 생성 (training 만) ----------
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

    # ---------- [2] N_Memb x K 완전탐색 → 단일 최적 조합 ----------
    print('\n[2] N_Memb x K 완전 격자탐색 (원논문 식 12, N_Memb 임의고정 없음)')
    Z, sets, M = build_linkage(boxes)
    all_candidates, final = full_grid_search(Z, sets, M)

    print(f'\n  {"N_Memb":>7} {"K*":>6} {"N_eff":>6} {"ECC":>8} {"rho_eff":>8}')
    for n_memb, K_star, N_eff, ecc, rho in all_candidates:
        mark = '  <-- 전체 최적' if (n_memb, K_star) == (final[0], final[1]) else ''
        print(f'  {n_memb:>7} {K_star:>6} {N_eff:>6} {ecc:>8.4f} {rho:>8.3f}{mark}')

    n_memb_star, K_star, N_eff_star, ecc_star, rho_star = final
    print(f'\n  ▶ 최종 채택: N_Memb*={n_memb_star}, K*={K_star} '
          f'(N_eff={N_eff_star}, ECC={ecc_star:.4f}, rho_eff={rho_star:.3f})')

    # ---------- [3],[4] 최적 조합에서 수축 + Hold-out ----------
    print(f'\n[3~4] (N_Memb*={n_memb_star}, K*={K_star}) 의 효과적 클러스터에 '
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
    first_result = None
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        D_orig = np.mean([boxes[i]['dbar'] for i in mem])

        lo0_, hi0_, lo, hi = shrink_stage3(lo0, hi0, D_orig, surrogate, d_func, rng)
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

    # ---------- 대표 클러스터 상세 ----------
    cid, lo0_, hi0_, lo, hi = first_result
    print(f'\n  [클러스터 {cid}] 단일 대표 운전구간 (통합 전 → 수축 후)')
    print(f'  {"변수":>10} {"통합(전)":>20} {"수축(후)":>20}')
    for p, nm in enumerate(feat_names):
        print(f'  {nm:>10} [{lo0_[p]:>8.1f},{hi0_[p]:>8.1f}]   '
              f'[{lo[p]:>8.1f},{hi[p]:>8.1f}]')

    print('\n완료.')


if __name__ == '__main__':
    main()
