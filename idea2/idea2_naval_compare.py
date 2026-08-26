"""
================================================================================
아이디어 2번 — Naval 데이터에서 Jaccard vs Tversky 비교
================================================================================
동일한 MRS-PRIM 박스에 유사도 공식만 바꿔 적용한다.
  · Jaccard: 원논문 기준선
  · Tversky: 아이디어 2 고정 가중치(W_SMALL=0.8, W_LARGE=0.2)
  · 추가로 W_SMALL=1.0에서 W_LARGE를 0~1로 스윕한다.

평가는 원논문 정의의 ECC를 사용하며, Tversky 효과의 전제인 박스 크기
비대칭이 실제로 존재하는지도 먼저 진단한다.
================================================================================
"""
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, '..', 'naval_propulsion_dataset.csv')
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache',
                         'stochastic_prim_journal', 'idea2')
CACHE_VERSION = 'naval-jaccard-tversky-v1'

RESPONSE_COLS = [
    'GT_Compressor_decay_state_coefficient',
    'GT_Turbine_decay_state_coefficient',
]
DROP_COLS = [
    'GT_Compressor_inlet_air_temp_T1',
    'GT_Compressor_inlet_air_pressure_P1',
    'Port_Propeller_Torque_Tp',
]

ALPHA_PEEL = 0.05
MIN_SUPPORT = 250
S_OPTIONS = (7, 9, 10)
T_PER_SIZE = 250
SEED_PRIM = 1

N_MEMB = 5
RHO_LIMIT = 0.6
K_GRID_STEP = 20

W_SMALL_PROPOSED = 0.8
W_LARGE_PROPOSED = 0.2
W_LARGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0]


def build_multi_desirability(df):
    """두 열화계수 모두 1에 가까울수록 좋은 LTB desirability."""
    d_list = []
    for col in RESPONSE_COLS:
        y = df[col].values.astype(float)
        lower, upper = float(y.min()), float(y.max())
        d = (y - lower) / (upper - lower) if upper > lower else np.ones_like(y)
        d_list.append(np.clip(d, 1e-6, 1.0))
    return np.exp(np.mean(np.log(np.column_stack(d_list)), axis=1))


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
            lo_q = np.quantile(xp, ALPHA_PEEL)
            hi_q = np.quantile(xp, 1 - ALPHA_PEEL)
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
    total = len(S_OPTIONS) * T_PER_SIZE
    done = 0
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
            done += 1
            if done % 100 == 0:
                print(f'    trial {done}/{total} | 박스 {len(boxes)}개', flush=True)
    return boxes


def cache_path(X, D):
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode())
    digest.update(repr((ALPHA_PEEL, MIN_SUPPORT, S_OPTIONS,
                        T_PER_SIZE, SEED_PRIM)).encode())
    for array in (X, D):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return os.path.join(CACHE_DIR, f'naval_boxes_{digest.hexdigest()[:16]}.npy')


def load_or_build_boxes(X, D):
    path = cache_path(X, D)
    if os.path.exists(path):
        boxes = list(np.load(path, allow_pickle=True))
        print(f'  [cache] 박스 {len(boxes)}개 로드: {os.path.basename(path)}')
        return boxes
    boxes = build_boxes(X, D)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, np.array(boxes, dtype=object), allow_pickle=True)
    print(f'  박스 {len(boxes)}개 생성 및 저장: {os.path.basename(path)}')
    return boxes


def intersection_matrix(boxes, n_observations):
    membership = np.zeros((len(boxes), n_observations), dtype=np.float32)
    for i, box in enumerate(boxes):
        membership[i, box['idx']] = 1.0
    return membership @ membership.T


def tversky_similarity(inter, support, w_small, w_large):
    only_i = support[:, None] - inter
    only_j = support[None, :] - inter
    small_only = np.minimum(only_i, only_j)
    large_only = np.maximum(only_i, only_j)
    denom = inter + w_small * small_only + w_large * large_only
    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom != 0)
    return (sim + sim.T) / 2


def make_linkage(sim):
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average')


def evaluate(Z, sets, K):
    M = len(sets)
    labels = fcluster(Z, K, criterion='maxclust')
    gammas, member_count, effective_count = [], 0, 0
    for cluster_id in np.unique(labels):
        members = np.where(labels == cluster_id)[0]
        if len(members) < N_MEMB:
            continue
        intersection = set.intersection(*[sets[m] for m in members])
        union = set.union(*[sets[m] for m in members])
        gammas.append(len(intersection) / len(union) if union else 0.0)
        member_count += len(members)
        effective_count += 1
    return {
        'ECC': float(np.mean(gammas)) if gammas else 0.0,
        'N_eff': effective_count,
        'rho': member_count / M,
    }


def find_kstar(Z, sets):
    M = len(sets)
    best = None
    for K in range(max(int(M * 0.05), 5), int(M * 0.50) + 1, K_GRID_STEP):
        result = evaluate(Z, sets, K)
        if result['rho'] >= RHO_LIMIT and (
                best is None or result['ECC'] > best[1]['ECC']):
            best = (K, result)
    return best


def diagnose_asymmetry(boxes, inter):
    support = np.array([box['support'] for box in boxes], dtype=float)
    upper = np.triu_indices(len(boxes), k=1)
    overlap = inter[upper]
    mask = overlap > 0
    small = np.minimum(support[upper[0]], support[upper[1]])[mask]
    large = np.maximum(support[upper[0]], support[upper[1]])[mask]
    overlap = overlap[mask]
    ratios = large / small
    contain = np.sum((overlap >= 0.8 * small) & (ratios >= 2.0))
    return {
        'support_min': int(support.min()),
        'support_max': int(support.max()),
        'support_ratio': float(support.max() / support.min()),
        'pair_ratio_median': float(np.median(ratios)) if len(ratios) else 1.0,
        'pair_ratio_max': float(ratios.max()) if len(ratios) else 1.0,
        'contain_pairs': int(contain),
        'overlap_pairs': int(mask.sum()),
    }


def summarize(name, best):
    if best is None:
        print(f'  {name:<24}: rho_Limit 만족 K 없음')
        return
    K, result = best
    print(f'  {name:<24}: K*={K:<4} ECC={result["ECC"]:.4f}  '
          f'N_eff={result["N_eff"]:<3} rho_eff={result["rho"]:.3f}')


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    feature_cols = [col for col in df.columns
                    if col not in RESPONSE_COLS and col not in DROP_COLS]
    X = df[feature_cols].values.astype(float)
    D = build_multi_desirability(df)
    print(f'Naval 데이터: {len(df)}행, 입력 {len(feature_cols)}개, 반응 2개')
    print(f'D 평균={D.mean():.4f}, 표준편차={D.std():.4f}')

    print('\nMRS-PRIM 박스 생성 ...')
    boxes = load_or_build_boxes(X, D)
    sets = [set(box['idx'].tolist()) for box in boxes]
    support = np.array([box['support'] for box in boxes], dtype=float)

    print('\n박스 교집합 행렬 계산 ...', flush=True)
    inter = intersection_matrix(boxes, len(df))
    diag = diagnose_asymmetry(boxes, inter)
    print(f'  support={diag["support_min"]}~{diag["support_max"]} '
          f'({diag["support_ratio"]:.2f}배)')
    print(f'  겹치는 쌍 크기비: 중앙 {diag["pair_ratio_median"]:.2f}배, '
          f'최대 {diag["pair_ratio_max"]:.2f}배')
    print(f'  비대칭 포함 쌍={diag["contain_pairs"]}/{diag["overlap_pairs"]} '
          f'({100 * diag["contain_pairs"] / max(diag["overlap_pairs"], 1):.2f}%)')

    sim_jaccard = tversky_similarity(inter, support, 1.0, 1.0)
    sim_proposed = tversky_similarity(
        inter, support, W_SMALL_PROPOSED, W_LARGE_PROPOSED)
    best_jaccard = find_kstar(make_linkage(sim_jaccard), sets)
    best_proposed = find_kstar(make_linkage(sim_proposed), sets)

    print('\n' + '=' * 72)
    print('고정 가중치 비교')
    print('=' * 72)
    summarize('Jaccard (원논문)', best_jaccard)
    summarize('Tversky (0.8, 0.2)', best_proposed)
    if best_jaccard and best_proposed:
        difference = best_proposed[1]['ECC'] - best_jaccard[1]['ECC']
        print(f'  ECC 차이: {difference:+.4f}')

    print('\n' + '=' * 72)
    print('Tversky W_LARGE 스윕 (W_SMALL=1.0)')
    print('=' * 72)
    print(f'  {"W_LARGE":>8} {"K*":>6} {"ECC":>8} {"N_eff":>7} '
          f'{"rho_eff":>9} {"vs Jaccard":>12}')
    sweep_results = []
    baseline_ecc = best_jaccard[1]['ECC']
    for w_large in W_LARGE_GRID:
        sim = tversky_similarity(inter, support, 1.0, w_large)
        best = find_kstar(make_linkage(sim), sets)
        if best is None:
            print(f'  {w_large:>8.1f}  (rho_Limit 미충족)')
            continue
        K, result = best
        difference = result['ECC'] - baseline_ecc
        sweep_results.append((w_large, K, result, difference))
        print(f'  {w_large:>8.1f} {K:>6} {result["ECC"]:>8.4f} '
              f'{result["N_eff"]:>7} {result["rho"]:>9.3f} {difference:>+12.4f}')

    best_weight = max(sweep_results, key=lambda item: item[2]['ECC'])
    print(f'\n최고 ECC: W_LARGE={best_weight[0]:.1f}, '
          f'ECC={best_weight[2]["ECC"]:.4f}, '
          f'Jaccard 대비 {best_weight[3]:+.4f}')
    print('\n완료.')


if __name__ == '__main__':
    main()
