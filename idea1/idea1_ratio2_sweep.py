"""
================================================================================
아이디어 1번 — 완전실측 2차 PRIM의 RATIO2(min_support 비율) 스윕
================================================================================
idea1_nested_prim.py 의 RATIO2(=0.5 고정)를 여러 값으로 스윕해 최적 지점을
탐색한다. 예측모델/몬테카를로는 전혀 쓰지 않으며(완전 실측 기반), 오직
2차 PRIM의 min_support(=n_box * RATIO2)만 바꿔가며 결과를 비교한다.

선택 기준: confirmation 실측 desirability(D_conf, 수축 후) 평균이 가장
높아지는 RATIO2 를 채택한다 ('선호도 점수가 높을 가능성이 높은 단일 범위
출력'이라는 목적에 맞춤). n_conf(표본 수)는 참고치로 함께 표시한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_master_train.npy')

TRAIN_RATIO = 0.7
SPLIT_SEED = 0

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 25

MIN_SUPPORT2_FLOOR = 20
RATIO2_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TOPN_CLUSTERS = 10
FEAT_NAMES = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
              'CoarseAgg', 'FineAgg', 'Age']


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


# ============================================= MRS-PRIM (1차: 박스 생성용, stochastic)
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
    print('  RATIO2 휴리스틱 스윕 (목적: D_conf 수축후 평균 최대화)')
    print('=' * 78)
    print(f'  {"RATIO2":>7} {"평균D_conf(후)":>15} {"평균개선폭":>11} '
          f'{"평균n_conf(후)":>15} {"n_conf=0 비율":>13}')

    # 클러스터별 통합박스/부분데이터셋은 RATIO2 와 무관하므로 미리 계산해 재사용
    precomputed = []
    for cid, mem in clusters:
        all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
        lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
        in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
        X_sub, D_sub = X_train[in_box], D_train[in_box]
        n_box = int(in_box.sum())
        d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)
        precomputed.append((cid, lo0, hi0, X_sub, D_sub, n_box, d_before, n_before))

    sweep_results = []
    for ratio2 in RATIO2_GRID:
        d_afters, diffs, n_afters, zero_count = [], [], [], 0
        for cid, lo0, hi0, X_sub, D_sub, n_box, d_before, n_before in precomputed:
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
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X_all = df.iloc[:, :8].values.astype(float)
    y_all = df.iloc[:, 8].values.astype(float)

    rng_split = np.random.default_rng(SPLIT_SEED)
    n = len(y_all)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_confirm = X_all[confirm_idx]

    d_func, lower, upper = make_desirability(y_all)
    D_train = d_func(y_train)
    D_confirm = d_func(y_all[confirm_idx])
    print(f'[0] 분할: training {len(y_train)}개 / confirmation {len(X_confirm)}개')

    print('\n[1] training 데이터로 MRS-PRIM 박스 생성')
    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드')
    else:
        boxes = build_boxes(X_train, D_train)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'  박스 {len(boxes)}개 생성 및 저장')

    print('\n[2] N_Memb x K 완전 격자탐색')
    Z, sets, M = build_linkage(boxes)
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

    # ---------- 채택된 RATIO2 로 최종 결과 상세 출력 ----------
    print(f'\n[3] 채택된 RATIO2={best_ratio2} 로 최종 수축 수행')
    print(f'\n  {"클러스터":>7} {"멤버":>4} {"n_box":>6} {"min_sup2":>8} {"D_conf(전)":>11} '
          f'{"D_conf(후)":>11} {"개선폭":>8} {"n_conf(전)":>10} {"n_conf(후)":>10}')
    diffs, improved = [], 0
    first_result = None
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

        if first_result is None:
            first_result = (cid, lo0, hi0, lo, hi)

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

    # ---------- 단일 대표 운전구간 출력 (첫 클러스터) ----------
    cid, lo0, hi0, lo, hi = first_result
    print(f'\n  [클러스터 {cid}] 단일 대표 운전구간 (통합 전 → 2차 PRIM 후)')
    print(f'  {"변수":>10} {"통합(전)":>20} {"2차 PRIM(후)":>20}')
    for p, nm in enumerate(FEAT_NAMES):
        print(f'  {nm:>10} [{lo0[p]:>8.1f},{hi0[p]:>8.1f}]   '
              f'[{lo[p]:>8.1f},{hi[p]:>8.1f}]')

    print('\n완료. (예측모델/몬테카를로 미사용, 전 과정 실측 기반)')


if __name__ == '__main__':
    main()
