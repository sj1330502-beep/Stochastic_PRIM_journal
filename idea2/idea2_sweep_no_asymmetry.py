"""
================================================================================
Tversky 가중치(W_LARGE) 스윕 — 비대칭이 없는 '표준 설정' 데이터에서
================================================================================
배경
  1~2차 실험(콘크리트/factory, 표준 min_support 고정)에서는 W_LARGE=0.2
  '한 값'만 확인하고 "효과 없음"이라 결론지었다. 이론적으로는 박스 크기가
  균일하면(비대칭 없음) 가중치를 얼마로 바꾸든 결과가 동일해야 하지만,
  실제로 스윕해서 확인한 적은 없었다. 이번 실험은 이를 직접 검증한다.

데이터
  콘크리트, 표준 설정(min_support=100 고정, 크기 다양화 없음) —
  1차 실험과 동일한 박스 집단(support 100~119, 사실상 균일 크기)을 사용.

방법
  W_SMALL=1.0 고정, W_LARGE 를 0.0~1.0 사이 촘촘히 스윕하며 ECC 비교.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_idea3_cache.npy')   # 표준 설정 박스(기존 캐시 재사용)

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1
N_MEMB, RHO_LIMIT = 5, 0.6

W_SMALL_FIXED = 1.0
W_LARGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


# ============================================= MRS-PRIM (표준 설정, 크기 다양화 없음)
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


# ============================================= 벡터화 유사도
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
        print(f'[cache] 표준설정 박스 {len(boxes)}개 로드')
    else:
        print('MRS-PRIM 박스 생성 (표준 설정, min_support=100 고정) ...')
        boxes = build_boxes(X, D)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'박스 {len(boxes)}개 생성 및 저장')

    sup = np.array([b['support'] for b in boxes], dtype=float)
    print(f'\n박스 support 범위: {int(sup.min())} ~ {int(sup.max())} '
          f'(최대/최소 = {sup.max()/sup.min():.2f}배)  ← 비대칭 없는 표준 설정')

    Mem = membership_matrix(boxes, N)
    inter = (Mem @ Mem.T).astype(float)

    # --- Jaccard 기준선(=W_LARGE=1.0과 동일) ---
    dist_j = np.clip(1.0 - inter / (sup[:, None] + sup[None, :] - inter), 0, 1.0)
    np.fill_diagonal(dist_j, 0.0)
    Z_j = linkage(squareform(dist_j, checks=False), method='average')
    K_j, r_j = find_kstar(Z_j, boxes)
    print(f'\nJaccard(원논문) 기준선: K*={K_j}, ECC={r_j["ECC"]:.4f}, '
          f'N_eff={r_j["N_eff"]}, rho_eff={r_j["rho"]:.3f}')

    print('\n' + '=' * 66)
    print('  Tversky 가중치(W_LARGE) 스윕 — 비대칭 없는 표준 설정 데이터')
    print('=' * 66)
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
        tag = '  <-- Jaccard와 동일 극단' if w_large == 1.0 else ''
        print(f'  {w_large:>8.2f} {K_star:>6} {r["ECC"]:>8.4f} {r["N_eff"]:>6} '
              f'{r["rho"]:>8.3f} {diff:>+12.4f}{tag}')

    print('\n' + '=' * 66)
    print('  판정')
    print('=' * 66)
    eccs = [r[2] for r in results]
    spread = max(eccs) - min(eccs)
    print(f'  W_LARGE 0~1 구간에서 ECC 변동폭 = {spread:.4f} '
          f'(최소 {min(eccs):.4f} ~ 최대 {max(eccs):.4f})')
    if spread < 1e-3:
        print('  ▶ 가중치를 바꿔도 ECC가 사실상 변하지 않음')
        print('    (비대칭이 없으니 가중치가 계산에 실질적 영향을 못 줌 — 이론과 일치)')
    else:
        print('  ▶ 비대칭이 없다고 진단됐음에도 가중치에 따라 ECC가 변동함 — 추가 분석 필요')

    print('\n완료.')


if __name__ == '__main__':
    main()
