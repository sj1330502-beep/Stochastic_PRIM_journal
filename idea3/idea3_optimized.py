"""
================================================================================
아이디어 3번 개선판: 신뢰도 가중 강도(gamma)를 자동 탐색
================================================================================
배경
  1차 검증(gamma=1.0 고정)에서는 표준 설정(원논문 min_support=100)에서
  신뢰도 가중이 ECC를 오히려 낮췄다(-0.0227). 이후 실험에서 gamma를
  약하게 줄수록 유리한 경향이 있음을 확인했다(크기 다양화 데이터 기준).

개선 방향
  데이터·박스 생성은 원논문 표준 설정 그대로 유지한다(인위적 크기
  다양화 없음). 대신 가중 반영 강도 gamma 를 고정하지 않고 격자로
  스윕해, 이 표준 데이터에서도 ECC를 개선하는 지점이 있는지 확인한다.
    d_weighted = d_jaccard / (w_m * w_m')^gamma
  gamma=0 은 원논문과 동일(가중치 없음), gamma가 커질수록 신뢰도 반영이
  강해진다.

산출
  gamma별 ECC 곡선을 리포트하고, baseline(gamma=0) 대비 개선되는
  최적 gamma를 자동으로 선택해 최종 결과로 제시한다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_idea3_cache.npy')   # 표준 설정 박스 (1차 검증과 동일)

# --- 박스 생성 (원논문 표준 설정 그대로, 크기 다양화 없음) ---
ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1

# --- 클러스터링 (원논문) ---
N_MEMB, RHO_LIMIT = 5, 0.6

# --- gamma 탐색 격자 (촘촘하게) ---
GAMMA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]


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


# ============================================= MRS-PRIM (원논문 표준 설정)
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
            if done % 200 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= 신뢰도
def reliability_weights(boxes):
    dbar = np.array([b['dbar'] for b in boxes], dtype=float)
    sup = np.array([b['support'] for b in boxes], dtype=float)

    def norm01(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)

    D_tilde, S_tilde = norm01(dbar), norm01(sup)
    w = np.sqrt(D_tilde * S_tilde)
    return np.clip(w, 1e-3, 1.0), dbar, sup


# ============================================= 유사도/거리
def membership_matrix(boxes, N):
    M = len(boxes)
    Mem = np.zeros((M, N), dtype=np.int32)
    for i, b in enumerate(boxes):
        Mem[i, b['idx']] = 1
    return Mem


def jaccard_distance(boxes, N):
    Mem = membership_matrix(boxes, N)
    inter = (Mem @ Mem.T).astype(float)
    sup = np.array([b['support'] for b in boxes], dtype=float)
    union = sup[:, None] + sup[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    sim = (sim + sim.T) / 2
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def apply_reliability_gamma(dist, w, gamma):
    if gamma == 0.0:
        return dist.copy()
    scale = (w[:, None] * w[None, :]) ** gamma
    d_weighted = dist / scale
    np.fill_diagonal(d_weighted, 0.0)
    return d_weighted


def make_linkage(dist):
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average')


# ============================================= ECC (원논문 정의)
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
        print(f'[cache] 박스 {len(boxes)}개 로드 (원논문 표준 설정)')
    else:
        print('MRS-PRIM 박스 생성 (원논문 표준 설정, min_support=100) ...')
        boxes = build_boxes(X, D)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)
        print(f'박스 {len(boxes)}개 생성 및 저장')

    w, dbar, sup = reliability_weights(boxes)
    print(f'\n신뢰도 편차: desirability CV={dbar.std()/dbar.mean():.3f}, '
          f'support CV={sup.std()/sup.mean():.3f}  (표준 설정 → 편차 작음, 참고용)')

    dist_base = jaccard_distance(boxes, N)

    print('\n' + '=' * 66)
    print('  gamma 자동 탐색 (표준 데이터, 크기 다양화 없음)')
    print('=' * 66)
    print(f'  {"gamma":>6} {"K*":>6} {"ECC":>8} {"N_eff":>6} '
          f'{"rho_eff":>8} {"ECC-baseline":>13}')

    results = []
    for gamma in GAMMA_GRID:
        dist_w = apply_reliability_gamma(dist_base, w, gamma)
        Z = make_linkage(dist_w)
        best = find_kstar(Z, boxes)
        if best is None:
            print(f'  {gamma:>6.2f}   (rho_Limit 미충족)')
            continue
        K_star, r = best
        results.append((gamma, K_star, r['ECC'], r['N_eff'], r['rho']))

    base_ecc = next(r[2] for r in results if r[0] == 0.0)
    for gamma, K_star, ecc, neff, rho in results:
        diff = ecc - base_ecc
        tag = '  <-- 원논문(가중치 없음)' if gamma == 0.0 else ''
        print(f'  {gamma:>6.2f} {K_star:>6} {ecc:>8.4f} {neff:>6} '
              f'{rho:>8.3f} {diff:>+13.4f}{tag}')

    # ---------- 자동 선택: baseline보다 나은 gamma 중 최고 ----------
    print('\n' + '=' * 66)
    print('  최종 결론 (자동 선택)')
    print('=' * 66)
    best_overall = max(results, key=lambda r: r[2])
    gamma_opt, K_opt, ecc_opt, neff_opt, rho_opt = best_overall
    print(f'  최적 gamma = {gamma_opt}')
    print(f'  K*={K_opt}, ECC={ecc_opt:.4f}, N_eff={neff_opt}, rho_eff={rho_opt:.3f}')
    print(f'  baseline(gamma=0) 대비 = {ecc_opt-base_ecc:+.4f}')
    if gamma_opt > 0.0 and (ecc_opt - base_ecc) > 1e-3:
        print('  ▶ 표준 데이터에서도 약한 신뢰도 가중이 소폭 개선을 준다')
    else:
        print('  ▶ 표준 데이터에서는 가중치 없음(gamma=0)이 최선이거나 차이 없음 —')
        print('    이 데이터는 신뢰도 편차 자체가 작아 가중의 이점이 제한적')

    print('\n완료.')


if __name__ == '__main__':
    main()
