"""
================================================================================
아이디어 5번 — 콘크리트 + DRS-PRIM 두 번째 케이스 스터디 (DRO)
================================================================================
목적: 원논문(제철, MRO/MRS-PRIM 단일 케이스)의 클러스터링 프레임워크가
      DRO(이중반응 최적화) 문제에서도 동일하게 작동하는지 보여준다.

구조
  [1부] DRS-PRIM: 콘크리트 데이터에 MSE=(mean-Target)^2+variance 최소화로
        박스 생성. 설정은 Koo et al. 원 논문의 콘크리트 DRO 실험 그대로
        (min_support=130, alpha=0.05, subspace S∈{5,6,7}, T=300/size).
  [2부] 클러스터링 프레임워크: 경험적 Jaccard -> average-linkage ->
        N_Memb x K 완전탐색으로 K* 확정 (박스 생성 방식과 무관하게 동일 로직).

이 두 부분이 문제없이 이어진다는 것 자체가 "프레임워크가 MRS/DRS 어느 쪽
박스에도 적용 가능한 범용 구조"라는 주장의 근거가 된다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')

# --- DRS-PRIM 설정 (Koo et al. 콘크리트 DRO 케이스와 동일) ---
TARGET = 60.0
ALPHA_PEEL = 0.05
MIN_SUPPORT = 130
S_OPTIONS = (5, 6, 7)
T_PER_SIZE = 300
SEED_PRIM = 1

# --- 클러스터링 완전탐색 (원논문 방식) ---
N_MEMB_OPTIONS = [5, 6, 7, 8, 9, 10]
RHO_LIMIT = 0.6
K_GRID_STEP = 20


# ============================================= [1부] DRS-PRIM (MSE 최소화)
def box_mse(y):
    """MSE = (평균 - Target)^2 + 분산  (Koo et al. 식 2, 실측만 사용)"""
    return (y.mean() - TARGET) ** 2 + y.var(ddof=1)


def peel_trajectory(X, y, S, rng):
    """단일 stochastic trial: MSE가 최소가 되는 방향으로 peeling"""
    P = X.shape[1]
    idx = np.arange(len(y))
    best_idx, best_obj = idx.copy(), box_mse(y)
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < MIN_SUPPORT:
            break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, np.inf
        for p in feats:
            xp = X[idx, p]
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
            for keep in (idx[xp > lo_q], idx[xp < hi_q]):
                if MIN_SUPPORT <= len(keep) < len(idx):
                    obj = box_mse(y[keep])
                    if obj < cand_obj:              # MSE 최소화 (desirability와 반대 방향)
                        cand_obj, cand_keep = obj, keep
        if cand_keep is None:
            break
        idx = cand_keep
        if cand_obj < best_obj:
            best_obj, best_idx = cand_obj, idx.copy()
    return best_idx, best_obj


def build_boxes_drs(X, y):
    rng = np.random.default_rng(SEED_PRIM)
    boxes, total, done = [], len(S_OPTIONS) * T_PER_SIZE, 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, y, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'mse': obj})
            done += 1
            if done % 300 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


# ============================================= [2부] 클러스터링 (원논문 방식, MRS와 동일 로직)
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
        for K in range(max(int(M * 0.05), 5), int(M * 0.5) + 1, K_GRID_STEP):
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
    return all_candidates, max(all_candidates, key=lambda c: c[3])


# ============================================= 실행
def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    feat_names = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplast',
                  'CoarseAgg', 'FineAgg', 'Age']
    print(f'데이터 {len(y)}행, 입력 8변수, 반응 1개(압축강도) — DRO 케이스\n')

    print('[1부] DRS-PRIM 박스 생성 (MSE 최소화, Target=60)')
    print(f'  min_support={MIN_SUPPORT}, S∈{S_OPTIONS}, 총 {len(S_OPTIONS)*T_PER_SIZE} trials')
    boxes = build_boxes_drs(X, y)
    mse_arr = np.array([b['mse'] for b in boxes])
    sup_arr = np.array([b['support'] for b in boxes])
    print(f'  박스 {len(boxes)}개 | MSE: min={mse_arr.min():.1f} '
          f'중앙={np.median(mse_arr):.1f} max={mse_arr.max():.1f} | '
          f'support: {sup_arr.min()}~{sup_arr.max()}\n')

    print('[2부] 클러스터링 프레임워크 (경험적 Jaccard -> average-linkage -> N_Memb x K 완전탐색)')
    Z, sets, M = build_linkage(boxes)
    all_candidates, final = full_grid_search(Z, sets, M)
    n_memb_star, K_star, N_eff, ecc, rho = final

    print(f'\n  {"N_Memb":>7} {"K*":>6} {"N_eff":>6} {"ECC":>8} {"rho_eff":>8}')
    for n_memb, K_s, neff, e, r in all_candidates:
        mark = '  <-- 전체 최적' if (n_memb, K_s) == (n_memb_star, K_star) else ''
        print(f'  {n_memb:>7} {K_s:>6} {neff:>6} {e:>8.4f} {r:>8.3f}{mark}')

    print(f'\n  ▶ 최종 채택: N_Memb*={n_memb_star}, K*={K_star} '
          f'(N_eff={N_eff}, ECC={ecc:.4f}, rho_eff={rho:.3f})')

    # 대표 클러스터(멤버 최다) 하나의 대표구간 출력 — 결과를 눈으로 확인하기 위함
    lab = fcluster(Z, K_star, criterion='maxclust')
    clusters = [(c, np.where(lab == c)[0]) for c in np.unique(lab)]
    clusters = [(c, m) for c, m in clusters if len(m) >= n_memb_star]
    clusters.sort(key=lambda x: -len(x[1]))
    cid, mem = clusters[0]
    all_idx = np.concatenate([boxes[i]['idx'] for i in mem])
    lo, hi = X[all_idx].min(axis=0), X[all_idx].max(axis=0)
    mean_mse = np.mean([boxes[i]['mse'] for i in mem])
    print(f'\n  [최대 클러스터 {cid}] 멤버 {len(mem)}개, 평균 MSE={mean_mse:.1f}')
    print(f'  {"변수":>10} {"구간":>20}')
    for p, nm in enumerate(feat_names):
        print(f'  {nm:>10} [{lo[p]:>8.1f}, {hi[p]:>8.1f}]')

    print('\n완료 — MRO(제철) 단일 케이스에 이어, DRO(콘크리트) 케이스에서도')
    print('동일한 클러스터링 프레임워크가 정상 작동함을 확인.')


if __name__ == '__main__':
    main()
