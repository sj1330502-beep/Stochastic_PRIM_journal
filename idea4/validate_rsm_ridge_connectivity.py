"""
================================================================================
RSM 고성능 영역 연결성 검증
================================================================================
목적
  synth_validation_sixth.py에서 적합한 RSM의 고성능 영역이 여러 개의 고립된
  봉우리가 아니라 하나의 넓게 이어진 영역인지 검증한다.

검증 절차
  1. 6차 방식으로 실제 콘크리트와 유사한 후보 배합을 대량 생성한다.
  2. desirability 임계값(D >= 0.80, 0.85, 0.90, 0.95)을 적용한다.
  3. 고성능 배합을 가까운 이웃끼리 연결한다.
  4. 두 배합 사이를 직선으로 세분화해, 경로 중간에서도 임계값 이상일 때만
     연결을 인정한다.
  5. 연결요소 수와 최대 연결요소 점유율을 계산한다.
  6. 실제 레시피 단위 bootstrap으로 RSM을 다시 적합하여 결론의 안정성을 본다.

해석
  - significant_components == 1
  - largest_component_share >= 0.90
  - bootstrap support_rate >= 0.80
  위 조건이 여러 threshold/radius에서 반복되면, 관측 범위와 RSM 안에서
  "하나의 넓게 연결된 고성능 영역"이라는 강한 증거로 해석할 수 있다.

주의
  이는 유한 표본과 학습된 RSM에 대한 수치적 검증이지 실제 물리공간 전체에
  대한 수학적 증명은 아니다.
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from synth_validation_sixth import (
    build_recipe_pool,
    desirability_from_strength,
    fit_response_surface,
    load_real_Xy,
    rsm_predict,
    sample_recipe_replicate_inputs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
DEFAULT_OUTPUT = os.path.join(HERE, 'rsm_ridge_connectivity_results.csv')

DEFAULT_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
DEFAULT_RADII = (0.04, 0.06, 0.08, 0.10)
DEFAULT_N_SAMPLES = 20000
DEFAULT_K_NEIGHBORS = 20
DEFAULT_PATH_POINTS = 11
DEFAULT_BOOTSTRAPS = 20

LARGEST_SHARE_CRITERION = 0.90
BOOTSTRAP_SUPPORT_CRITERION = 0.80
SIGNIFICANT_COMPONENT_FRACTION = 0.01
SIGNIFICANT_COMPONENT_MIN_SIZE = 5


def normalized_coordinates(X, lo, hi):
    """변수 범위를 보정한 RMS 거리 좌표."""
    span = np.where(hi > lo, hi - lo, 1.0)
    return ((X - lo) / span) / np.sqrt(X.shape[1])


def unique_knn_edges(coords, k_neighbors, max_radius):
    """중복 없는 k-NN 후보 간선을 반환한다."""
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    k_query = min(k_neighbors + 1, n)
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=k_query)
    distances = np.asarray(distances)
    neighbors = np.asarray(neighbors)

    if distances.ndim == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]

    rows = np.repeat(np.arange(n), k_query - 1)
    cols = neighbors[:, 1:].reshape(-1)
    dists = distances[:, 1:].reshape(-1)
    valid = np.isfinite(dists) & (dists <= max_radius) & (rows != cols)

    a = np.minimum(rows[valid], cols[valid])
    b = np.maximum(rows[valid], cols[valid])
    if len(a) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    encoded = a.astype(np.int64) * np.int64(n) + b.astype(np.int64)
    _, first = np.unique(encoded, return_index=True)
    edges = np.column_stack((a[first], b[first]))
    return edges, dists[valid][first]


def minimum_desirability_along_edges(model, X, D, edges, y_min, y_max,
                                     path_points, batch_size=2000):
    """각 간선을 세분화하여 경로상 최소 desirability를 구한다."""
    if len(edges) == 0:
        return np.empty(0, dtype=float)

    t = np.linspace(0.0, 1.0, path_points)
    minimum = np.empty(len(edges), dtype=float)

    for start in range(0, len(edges), batch_size):
        stop = min(start + batch_size, len(edges))
        batch = edges[start:stop]
        x0 = X[batch[:, 0]]
        x1 = X[batch[:, 1]]
        points = x0[:, None, :] + t[None, :, None] * (
            x1[:, None, :] - x0[:, None, :]
        )
        y_path = rsm_predict(model, points.reshape(-1, X.shape[1]))
        d_path = desirability_from_strength(y_path, y_min, y_max)
        d_path = d_path.reshape(len(batch), path_points)

        # 계산 오차에 대비해 이미 계산된 양 끝점의 D도 포함한다.
        d_path[:, 0] = D[batch[:, 0]]
        d_path[:, -1] = D[batch[:, 1]]
        minimum[start:stop] = d_path.min(axis=1)

    return minimum


def prepare_threshold_edges(X, D, model, y_min, y_max, threshold,
                            k_neighbors, max_radius, path_points, lo, hi):
    """하나의 D 임계값에서 거리 및 경로 검증을 마친 후보 간선을 준비한다."""
    original_indices = np.where(D >= threshold)[0]
    X_high = X[original_indices]
    D_high = D[original_indices]
    coords = normalized_coordinates(X_high, lo, hi)

    edges, distances = unique_knn_edges(coords, k_neighbors, max_radius)
    edge_min_D = minimum_desirability_along_edges(
        model, X_high, D_high, edges, y_min, y_max, path_points
    )
    return X_high, D_high, coords, edges, distances, edge_min_D


def make_adjacency(n, edges, distances, valid):
    """유효 간선으로 무방향 가중 인접행렬을 생성한다."""
    kept_edges = edges[valid]
    kept_distances = distances[valid]
    if len(kept_edges) == 0:
        return csr_matrix((n, n), dtype=float)

    rows = np.concatenate((kept_edges[:, 0], kept_edges[:, 1]))
    cols = np.concatenate((kept_edges[:, 1], kept_edges[:, 0]))
    weights = np.concatenate((kept_distances, kept_distances))
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def recover_path(predecessors, source, target):
    """dijkstra predecessor 배열에서 source→target 경로를 복원한다."""
    path = [int(target)]
    current = int(target)
    while current != source:
        current = int(predecessors[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def distant_path_diagnostic(coords, adjacency, labels, largest_label):
    """최대 연결요소 안에서 멀리 떨어진 두 점과 연결 경로를 진단한다."""
    members = np.where(labels == largest_label)[0]
    if len(members) < 2:
        return {
            'endpoint_distance': np.nan,
            'path_length': np.nan,
            'path_hops': 0,
        }

    seed = members[0]
    first = members[np.argmax(np.linalg.norm(coords[members] - coords[seed], axis=1))]
    second = members[np.argmax(np.linalg.norm(coords[members] - coords[first], axis=1))]

    distances, predecessors = dijkstra(
        adjacency, directed=False, indices=first, return_predecessors=True
    )
    path = recover_path(predecessors, int(first), int(second))
    return {
        'endpoint_distance': float(np.linalg.norm(coords[first] - coords[second])),
        'path_length': float(distances[second]) if np.isfinite(distances[second]) else np.nan,
        'path_hops': max(len(path) - 1, 0),
    }


def summarize_graph(coords, edges, distances, edge_min_D, threshold, radius,
                    total_candidates):
    """하나의 threshold/radius 조합에서 연결성 지표를 계산한다."""
    n = len(coords)
    if n == 0:
        return {
            'n_high': 0,
            'high_fraction': 0.0,
            'n_edges': 0,
            'components': 0,
            'significant_components': 0,
            'largest_component_size': 0,
            'largest_component_share': 0.0,
            'isolated_share': 1.0,
            'endpoint_distance': np.nan,
            'path_length': np.nan,
            'path_hops': 0,
            'connected_region_supported': False,
        }

    valid = (distances <= radius) & (edge_min_D >= threshold - 1e-12)
    adjacency = make_adjacency(n, edges, distances, valid)
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    largest_label = int(np.argmax(sizes))
    largest_size = int(sizes[largest_label])
    largest_share = largest_size / n
    isolated_share = float(np.mean(np.asarray(adjacency.getnnz(axis=1)).ravel() == 0))

    significant_min = max(
        SIGNIFICANT_COMPONENT_MIN_SIZE,
        int(np.ceil(SIGNIFICANT_COMPONENT_FRACTION * n)),
    )
    significant_components = int(np.sum(sizes >= significant_min))
    path_diag = distant_path_diagnostic(coords, adjacency, labels, largest_label)

    supported = (
        significant_components == 1
        and largest_share >= LARGEST_SHARE_CRITERION
    )
    return {
        'n_high': n,
        'high_fraction': n / total_candidates,
        'n_edges': int(valid.sum()),
        'components': int(n_components),
        'significant_components': significant_components,
        'largest_component_size': largest_size,
        'largest_component_share': largest_share,
        'isolated_share': isolated_share,
        **path_diag,
        'connected_region_supported': bool(supported),
    }


def connectivity_sweep(X, model, y_min, y_max, thresholds, radii,
                       k_neighbors, path_points, lo, hi, replicate):
    """한 RSM에 대해 모든 threshold/radius 조합을 평가한다."""
    y_pred = rsm_predict(model, X)
    D = desirability_from_strength(y_pred, y_min, y_max)
    max_radius = max(radii)
    rows = []

    for threshold in thresholds:
        prepared = prepare_threshold_edges(
            X, D, model, y_min, y_max, threshold,
            k_neighbors, max_radius, path_points, lo, hi,
        )
        _, _, coords, edges, distances, edge_min_D = prepared

        for radius in radii:
            result = summarize_graph(
                coords, edges, distances, edge_min_D,
                threshold, radius, len(X),
            )
            result.update({
                'replicate': replicate,
                'threshold': threshold,
                'radius': radius,
                'rsm_r2': model['r2'],
            })
            rows.append(result)

    return rows


def recipe_cluster_bootstrap_indices(real_X, rng):
    """반복측정 의존성을 보존하는 레시피 단위 cluster bootstrap."""
    _, recipe_row_idx, _ = build_recipe_pool(real_X)
    sampled = rng.integers(0, len(recipe_row_idx), size=len(recipe_row_idx))
    return np.concatenate([np.asarray(recipe_row_idx[i], dtype=int) for i in sampled])


def bootstrap_sweep(real_X, real_y, candidate_X, thresholds, radii,
                    k_neighbors, path_points, n_bootstraps, seed):
    """레시피 단위 bootstrap으로 RSM 불확실성에 대한 안정성을 평가한다."""
    rng = np.random.default_rng(seed)
    lo, hi = real_X.min(axis=0), real_X.max(axis=0)
    rows = []

    base_model = fit_response_surface(real_X, real_y)
    rows.extend(connectivity_sweep(
        candidate_X, base_model, real_y.min(), real_y.max(),
        thresholds, radii, k_neighbors, path_points, lo, hi,
        replicate=0,
    ))

    for replicate in range(1, n_bootstraps + 1):
        idx = recipe_cluster_bootstrap_indices(real_X, rng)
        model = fit_response_surface(real_X[idx], real_y[idx])
        rows.extend(connectivity_sweep(
            candidate_X, model, real_y.min(), real_y.max(),
            thresholds, radii, k_neighbors, path_points, lo, hi,
            replicate=replicate,
        ))
        print(f'  bootstrap {replicate}/{n_bootstraps} 완료', flush=True)

    return pd.DataFrame(rows)


def print_base_results(results):
    """원본 RSM의 연결성 결과를 표로 출력한다."""
    base = results[results['replicate'] == 0].copy()
    columns = [
        'threshold', 'radius', 'n_high', 'significant_components',
        'largest_component_share', 'isolated_share',
        'endpoint_distance', 'path_hops', 'connected_region_supported',
    ]
    print('\n원본 RSM 연결성 결과')
    print(base[columns].to_string(index=False, formatters={
        'largest_component_share': '{:.3f}'.format,
        'isolated_share': '{:.3f}'.format,
        'endpoint_distance': '{:.3f}'.format,
    }))


def print_bootstrap_summary(results, n_bootstraps):
    """bootstrap 반복에서 연결 영역 판정이 유지되는 비율을 출력한다."""
    if n_bootstraps == 0:
        return

    boot = results[results['replicate'] > 0]
    summary = boot.groupby(['threshold', 'radius']).agg(
        support_rate=('connected_region_supported', 'mean'),
        mean_largest_share=('largest_component_share', 'mean'),
        sd_largest_share=('largest_component_share', 'std'),
        median_significant_components=('significant_components', 'median'),
        successful_runs=('connected_region_supported', 'sum'),
        total_runs=('connected_region_supported', 'size'),
    ).reset_index()
    summary['robust_support'] = (
        summary['support_rate'] >= BOOTSTRAP_SUPPORT_CRITERION
    )

    print('\nBootstrap 안정성 요약')
    print(summary.to_string(index=False, formatters={
        'support_rate': '{:.3f}'.format,
        'mean_largest_share': '{:.3f}'.format,
        'sd_largest_share': '{:.3f}'.format,
    }))


def parse_float_list(value):
    return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='RSM 고성능 영역이 하나로 연결되는지 수치적으로 검증합니다.'
    )
    parser.add_argument('--csv', default=CSV_PATH)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument('--thresholds', type=parse_float_list,
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument('--radii', type=parse_float_list, default=DEFAULT_RADII)
    parser.add_argument('--k-neighbors', type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument('--path-points', type=int, default=DEFAULT_PATH_POINTS)
    parser.add_argument('--bootstraps', type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument('--seed', type=int, default=20260826)
    return parser.parse_args()


def validate_args(args):
    if args.n_samples < 100:
        raise ValueError('--n-samples는 100 이상이어야 합니다.')
    if args.k_neighbors < 1:
        raise ValueError('--k-neighbors는 1 이상이어야 합니다.')
    if args.path_points < 2:
        raise ValueError('--path-points는 2 이상이어야 합니다.')
    if args.bootstraps < 0:
        raise ValueError('--bootstraps는 0 이상이어야 합니다.')
    if not args.thresholds or any(not 0 < x < 1 for x in args.thresholds):
        raise ValueError('--thresholds는 0과 1 사이 값이어야 합니다.')
    if not args.radii or any(x <= 0 for x in args.radii):
        raise ValueError('--radii는 양수여야 합니다.')


def main():
    args = parse_args()
    validate_args(args)

    real_X, real_y = load_real_Xy(args.csv)
    candidate_X = sample_recipe_replicate_inputs(
        real_X, args.n_samples, seed=args.seed
    )

    print(f'실측 데이터: {len(real_X)}개')
    print(f'연결성 후보 합성 배합: {len(candidate_X)}개')
    print(f'D 임계값: {args.thresholds}')
    print(f'연결 반경: {args.radii}')
    print(f'레시피 bootstrap: {args.bootstraps}회\n')

    results = bootstrap_sweep(
        real_X, real_y, candidate_X,
        thresholds=args.thresholds,
        radii=args.radii,
        k_neighbors=args.k_neighbors,
        path_points=args.path_points,
        n_bootstraps=args.bootstraps,
        seed=args.seed + 1,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(args.output, index=False, encoding='utf-8-sig')

    print_base_results(results)
    print_bootstrap_summary(results, args.bootstraps)
    print(f'\n상세 결과 저장: {args.output}')
    print('\n주의: 이 결과는 RSM과 관측된 입력 범위 안에서의 수치적 증거이며,')
    print('실제 콘크리트 물리공간 전체에 대한 수학적 증명은 아닙니다.')


if __name__ == '__main__':
    main()
