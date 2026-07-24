"""
================================================================================
Tversky 가중치(W_SMALL, W_LARGE) 스윕 실험 — 어느 지점에서 Jaccard를 이기나
================================================================================
배경
  이전 실험(idea2_varied_size.py)에서 min_support를 다단계로 섞어 실제
  25배 크기 비대칭을 만들었으나, W_SMALL=0.8/W_LARGE=0.2 고정값에서는
  Tversky가 오히려 Jaccard보다 근소하게 낮은 ECC를 보였다.

이번 실험
  같은 박스 집단(min_support 다단계)을 그대로 재사용하고,
  W_LARGE 값만 여러 단계로 바꿔가며 ECC가 어떻게 변하는지 관찰한다.
    W_LARGE ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0}
    (W_LARGE=1.0 이면 Jaccard와 동일한 극단, W_LARGE=0 이면 overlap
     coefficient에 가까운 극단)
  W_SMALL은 1.0으로 고정(작은 박스 차이는 항상 최대로 벌점 — Tversky의
  핵심 취지인 "포함관계는 살리되 남남은 그대로 벌준다"를 유지).

  박스 생성은 1회만 수행(캐시)하고, 유사도·K*·ECC만 가중치별로 재계산한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_varied_cache.npy')

ALPHA_PEEL = 0.05
SEED_PRIM = 1
N_MEMB, RHO_LIMIT = 5, 0.6

MIN_SUPPORT_TIERS = [30, 60, 120, 250, 500, 700]
S_OPTIONS = (4, 5, 6)
T_PER_TIER = 120

W_SMALL_FIXED = 1.0
W_LARGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


# ============================================= desirability
def desirability_ntb(y, target=60.0):
    y = np.asarray(y, dtype=float)
    lower, upper = float(y.min()), float(y.max())
    d = np.zeros_like(y)
    m1 = (y >= lower) & (y <= target)
    d[m1] = (y[m1] - lower) / (target - lower)
    m2 = (y > target) & (y <= upper)
    d[m2] = (upper - y[m2]) / (upper - target)
    return np.clip(d, 0.0, 1.0)


# ============================================= MRS-PRIM (min_support 가변)
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


def build_boxes_varied(X, D):
    rng = np.random.default_rng(SEED_PRIM)
    boxes = []
    total = len(MIN_SUPPORT_TIERS) * len(S_OPTIONS) * T_PER_TIER
    done = 0
    for min_sup in MIN_SUPPORT_TIERS:
        for S in S_OPTIONS:
            for _ in range(T_PER_TIER):
                idx, obj = peel_trajectory(X, D, S, min_sup, rng)
                if len(idx) >= min_sup:
                    boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
                done += 1
                if done % 400 == 0:
                    print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 벡터화 유사도 (가중치 파라미터화)
def membership_matrix(boxes, N):
    M = len(boxes)
    Mem = np.zeros((M, N), dtype=np.int32)
    for i, b in enumerate(boxes):
        Mem[i, b['idx']] = 1
    return Mem


def tversky_sim(inter, sup, w_small, w_large):
    a, b = sup[:, None], sup[None, :]
    only_i, only_j = a - inter, b - inter
    small_only = np.minimum(only_i, only_j)
    large_only = np.maximum(only_i, only_j)
    denom = inter + w_small * small_only + w_large * large_only
    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom != 0)
    return (sim + sim.T) / 2


def make_linkage(sim):
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average')


# ============================================= ECC
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
    return dict(ECC=float(np.mean(gams)) if gams else 0.0, N_eff=neff, rho=Meff / M)


def find_kstar(Z, boxes):
    M = len(boxes)
    best = None
    for K in range(max(int(M * 0.05), 5), int(M * 0.50) + 1, 15):
        r = evaluate(Z, boxes, K)
        if r['rho'] >= RHO_LIMIT and (best is None or r['ECC'] > best[1]['ECC']):
            best = (K, r)
    return best


# ============================================= 실행
def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    D = desirability_ntb(y)
    N = len(y)

    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
        print(f'[cache] 박스 {len(boxes)}개 로드')
    else:
        print('박스 생성 (min_support 다단계 혼합) ...')
        boxes = build_boxes_varied(X, D)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'박스 {len(boxes)}개 생성 및 저장')

    sup = np.array([b['support'] for b in boxes], dtype=float)
    print(f'support 범위: {int(sup.min())} ~ {int(sup.max())} '
          f'(최대/최소 = {sup.max()/sup.min():.2f}배)\n')

    Mem = membership_matrix(boxes, N)
    inter = (Mem @ Mem.T).astype(float)

    # --- Jaccard 기준선 ---
    dist_j = np.clip(1.0 - inter / (sup[:, None] + sup[None, :] - inter), 0, 1.0)
    np.fill_diagonal(dist_j, 0.0)
    Z_j = linkage(squareform(dist_j, checks=False), method='average')
    K_j, r_j = find_kstar(Z_j, boxes)
    print('=' * 66)
    print(f'  JACCARD (원논문, 기준선) : K*={K_j:<5} ECC={r_j["ECC"]:.4f}  '
          f'N_eff={r_j["N_eff"]:<4} rho_eff={r_j["rho"]:.3f}')
    print('=' * 66)

    # --- Tversky: W_LARGE 스윕 ---
    print(f'\nTversky 스윕 (W_SMALL={W_SMALL_FIXED} 고정, W_LARGE 변화)')
    print(f'  {"W_LARGE":>8} {"K*":>6} {"ECC":>8} {"N_eff":>6} '
          f'{"rho_eff":>8} {"ECC-Jaccard":>12}')
    results = []
    for w_large in W_LARGE_GRID:
        sim = tversky_sim(inter, sup, W_SMALL_FIXED, w_large)
        Z = make_linkage(sim)
        best = find_kstar(Z, boxes)
        if best is None:
            print(f'  {w_large:>8.2f}   (rho_Limit 미충족)')
            continue
        K_star, r = best
        diff = r['ECC'] - r_j['ECC']
        results.append((w_large, K_star, r['ECC'], r['N_eff'], r['rho'], diff))
        mark = '  <-- Jaccard와 동일 극단' if w_large == 1.0 else ''
        print(f'  {w_large:>8.2f} {K_star:>6} {r["ECC"]:>8.4f} {r["N_eff"]:>6} '
              f'{r["rho"]:>8.3f} {diff:>+12.4f}{mark}')

    # --- 최적 지점 ---
    if results:
        best_w = max(results, key=lambda x: x[2])
        print(f'\n최고 ECC 지점: W_LARGE={best_w[0]}, ECC={best_w[2]:.4f} '
              f'(Jaccard 대비 {best_w[5]:+.4f})')
        if best_w[5] > 1e-3:
            print('▶ 이 W_LARGE에서 Tversky가 Jaccard보다 우수함이 확인됨')
        else:
            print('▶ 스윕 범위 내에서 Jaccard를 유의미하게 능가하는 지점 없음')

    print('\n완료.')


if __name__ == '__main__':
    main()
