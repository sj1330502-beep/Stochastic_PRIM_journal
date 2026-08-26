"""
================================================================================
아이디어 1번 — 방법A 최종판: 몬테카를로 가상포인트 + 대리모델 기반 수축
================================================================================
effective cluster들 중 Score=sqrt(평균 desirability × compactness)가 가장 높은
클러스터를 recipe 후보로 선택한 뒤, 통합박스 경계면에 가상포인트를 몬테카를로로 뿌려
멤버 박스들과의 다차원 중첩 밀도가 가장 낮은 경계부터 깎아 들어간다.
품질 재평가(ΔD)는 대리모델(RandomForest) 예측으로 수행한다.

[최종 개선사항]
  1. 종료 조건을 '부피 하한'에서 '남은 관측치 비율 하한(N_MIN_RATIO)'으로
     교체 -- 부피만으로는 실제 표본 밀도를 반영하지 못해 confirmation
     표본이 과도하게 줄어드는 문제가 있었다.
  2. ΔD 평가용 가상포인트(10,000개)와 대리모델 예측을 매 반복 재추출하지
     않고 최초 1회만 계산 후, 이후에는 현재 박스 안에 남은 점만 필터링해
     재사용한다 (박스가 단조 수축하므로 상위집합을 계속 거르는 것으로 충분).

10개의 서로 다른 랜덤 training/confirmation 분할로 반복 검증한다.
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

TRAIN_RATIO = 0.7
N_RANDOM_SEEDS = 10

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 25

MC_SAMPLES, ALPHA_Q = 10000, 5
DELTA, BAND = 0.05, 0.10
EPS, MAX_ITER = 0.05, 300
N_MIN_RATIO = 0.4


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
    return d_func


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


def build_boxes(X, D, seed):
    rng = np.random.default_rng(seed)
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
    return boxes


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
        return None
    return max(all_candidates, key=lambda c: c[3])


def eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0):
    mask = np.all((pts0 >= lo) & (pts0 <= hi), axis=1)
    n_pts = int(mask.sum())
    if n_pts < 20:
        return np.inf, n_pts
    dD = D_orig - np.percentile(dpred0[mask], ALPHA_Q)
    return dD, n_pts


def virtual_overlap_density(lo, hi, member_los, member_his, p, face, rng, n_probe=2000):
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
    covered = np.zeros(n_probe, dtype=bool)
    for m_lo, m_hi in zip(member_los, member_his):
        covered |= np.all((pts >= m_lo) & (pts <= m_hi), axis=1)
    return covered.mean()


def shrink_methodA(lo0, hi0, D_orig, member_los, member_his, surrogate, d_func, rng, X_train):
    lo, hi = lo0.copy(), hi0.copy()
    P = len(lo)
    n0 = int(np.sum(np.all((X_train >= lo0) & (X_train <= hi0), axis=1)))
    n_min = max(1, int(n0 * N_MIN_RATIO))

    pts0 = rng.uniform(lo0, hi0, size=(MC_SAMPLES, len(lo0)))
    dpred0 = d_func(surrogate.predict(pts0))

    dD, n_pts = eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0)

    reason, n_iter = 'MAX_ITER', 0
    for it in range(MAX_ITER):
        n_iter = it
        n_remaining = int(np.sum(np.all((X_train >= lo) & (X_train <= hi), axis=1)))
        if dD <= EPS:
            reason = 'dD<=EPS'
            break
        if n_remaining <= n_min:
            reason = 'n<=n_min'
            break
        best_face, best_density = None, np.inf
        for p in range(P):
            for face in ('lo', 'hi'):
                dens = virtual_overlap_density(lo, hi, member_los, member_his, p, face, rng)
                if dens < best_density:
                    best_density, best_face = dens, (p, face)
        if best_face is None or not np.isfinite(best_density):
            reason = 'no_valid_face'
            break
        p, face = best_face
        step = DELTA * (hi[p] - lo[p])
        if face == 'lo':
            lo[p] = min(lo[p] + step, hi[p] - 1e-9)
        else:
            hi[p] = max(hi[p] - step, lo[p] + 1e-9)
        dD, n_pts = eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0)
    else:
        reason = 'MAX_ITER'
        n_iter = MAX_ITER

    n_remaining_final = int(np.sum(np.all((X_train >= lo) & (X_train <= hi), axis=1)))
    return lo, hi, reason, n_iter, dD, n_remaining_final, n_min, n0


def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float('nan')
    return d_mean, n


def select_recipe_cluster(labels, boxes, sets, n_memb_star):
    """
    Effective cluster 중 최종 recipe 후보를 선택한다.

    기존 방식: 멤버 subregion 수가 가장 많은 cluster 선택
    개선 방식:
        cluster_mean_D = 멤버 subregion들의 dbar 평균
        compactness   = 모든 멤버 subregion 관측치의 교집합 / 합집합
        score         = sqrt(cluster_mean_D * compactness)

    N_Memb 조건은 그대로 유지하므로, 박스 수는 effective 여부를 판단하는
    최소 신뢰성 조건으로 사용하고 최종 순위에는 직접 넣지 않는다.
    """
    candidates = []
    for cid in np.unique(labels):
        mem = np.where(labels == cid)[0]
        if len(mem) < n_memb_star:
            continue

        inter_set = set.intersection(*[sets[i] for i in mem])
        union_set = set.union(*[sets[i] for i in mem])
        compactness = len(inter_set) / len(union_set) if union_set else 0.0
        mean_d = float(np.mean([boxes[i]['dbar'] for i in mem]))
        score = float(np.sqrt(max(mean_d, 0.0) * max(compactness, 0.0)))

        candidates.append({
            'cid': int(cid),
            'mem': mem,
            'n_members': int(len(mem)),
            'mean_d': mean_d,
            'compactness': compactness,
            'score': score,
        })

    if not candidates:
        return None

    # 동점일 경우 mean D -> compactness -> member 수 순으로 결정
    return max(
        candidates,
        key=lambda c: (c['score'], c['mean_d'], c['compactness'], c['n_members'])
    )


def run_one_seed(df, split_seed):
    X_all = df.iloc[:, :8].values.astype(float)
    y_all = df.iloc[:, 8].values.astype(float)

    rng_split = np.random.default_rng(split_seed)
    n = len(y_all)
    perm = rng_split.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_confirm = X_all[confirm_idx]

    d_func = make_desirability(y_all)
    D_train = d_func(y_train)
    D_confirm = d_func(y_all[confirm_idx])

    boxes = build_boxes(X_train, D_train, split_seed)
    if len(boxes) < 10:
        return None
    Z, sets, M = build_linkage(boxes)
    final = full_grid_search(Z, sets, M)
    if final is None:
        return None
    n_memb_star, K_star, N_eff, ecc, rho = final

    lab = fcluster(Z, K_star, criterion='maxclust')
    selected = select_recipe_cluster(lab, boxes, sets, n_memb_star)
    if selected is None:
        return None
    cid, mem = selected['cid'], selected['mem']
    cluster_mean_d = selected['mean_d']
    cluster_compactness = selected['compactness']
    cluster_score = selected['score']
    cluster_n_members = selected['n_members']

    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    lo0, hi0 = X_train[all_idx].min(axis=0), X_train[all_idx].max(axis=0)
    D_orig = np.mean([boxes[i]['dbar'] for i in mem])
    d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)

    surrogate = RandomForestRegressor(n_estimators=100, random_state=split_seed
                                      ).fit(X_train, y_train)
    rng = np.random.default_rng(split_seed + 500000)
    member_los = np.array([X_train[boxes[i]['idx']].min(axis=0) for i in mem])
    member_his = np.array([X_train[boxes[i]['idx']].max(axis=0) for i in mem])
    lo_a, hi_a, reason_a, n_iter_a, dD_a, n_remaining_a, n_min_a, n0_a = shrink_methodA(
        lo0, hi0, D_orig, member_los, member_his, surrogate, d_func, rng, X_train)
    d_after_a, n_after_a = holdout_eval(lo_a, hi_a, X_confirm, D_confirm)

    return dict(split_seed=split_seed, K_star=K_star, n_memb_star=n_memb_star,
               cluster_id=cid, cluster_n_members=cluster_n_members,
               cluster_mean_d=cluster_mean_d,
               cluster_compactness=cluster_compactness, cluster_score=cluster_score,
               d_before=d_before, n_before=n_before,
               d_after_a=d_after_a, n_after_a=n_after_a,
               reason_a=reason_a, n_iter_a=n_iter_a, dD_a=dD_a,
               n_remaining_a=n_remaining_a, n_min_a=n_min_a, n0_a=n0_a)


def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

    SPLIT_SEEDS = [int(x) for x in
                  np.random.default_rng().integers(0, 1_000_000, size=N_RANDOM_SEEDS)]
    print(f'이번 실행에 사용할 랜덤 시드: {SPLIT_SEEDS}\n')

    print('=' * 90)
    print('  방법A(Score 기반 클러스터 선택 + 몬테카를로+대리모델) -- 랜덤 10개 분할 검증')
    print('=' * 90)

    results = []
    for seed in SPLIT_SEEDS:
        print(f'\n[분할 시드={seed}] 실행 중 ...')
        r = run_one_seed(df, seed)
        if r is None:
            print('  실패, 건너뜀')
            continue
        results.append(r)
        print(f'  선택 클러스터    : id={r["cluster_id"]}, members={r["cluster_n_members"]}, '
              f'meanD={r["cluster_mean_d"]:.4f}, compactness={r["cluster_compactness"]:.4f}, '
              f'Score={r["cluster_score"]:.4f}')
        print(f'  통합박스(수축 전): D_conf={r["d_before"]:.4f} (n={r["n_before"]})')
        print(f'  방법A(수축 후)   : D_conf={r["d_after_a"]:.4f} (n={r["n_after_a"]}) '
              f'| 종료사유={r["reason_a"]}, 반복={r["n_iter_a"]}회, '
              f'ΔD={r["dD_a"]:.4f}, 남은관측치={r["n_remaining_a"]}/{r["n0_a"]} '
              f'(하한={r["n_min_a"]})')

    print('\n' + '=' * 90)
    print('  종합 (분할 시드별 요약)')
    print('=' * 90)
    diffs = []
    for r in results:
        d = r['d_after_a'] - r['d_before'] if not np.isnan(r['d_before']) else float('nan')
        if not np.isnan(d):
            diffs.append(d)
        print(f'  시드={r["split_seed"]:>8}: 수축전={r["d_before"]:.4f} -> '
              f'수축후={r["d_after_a"]:.4f} (차이 {d:+.4f})')

    if diffs:
        print(f'\n  평균 개선폭 = {np.mean(diffs):+.4f} (표준편차 {np.std(diffs):.4f})')

    print('\n완료.')


if __name__ == '__main__':
    main()
