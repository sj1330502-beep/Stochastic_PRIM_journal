"""
================================================================================
MRS-PRIM 과분할 통제 합성실험
================================================================================
목적
  정답 구조를 미리 아는 두 합성 통제군에서 원 논문의 MRS-PRIM 및 클러스터링
  파이프라인이 하나의 고성능 영역을 여러 effective cluster로 나누는지 확인한다.

통제군
  A. connected: 하나의 넓고 연결된 고성능 slab (정답 component 1개)
  B. separated: 저성능 gap으로 분리된 세 slab (정답 component 3개)

두 통제군은 같은 합성 입력 X와 같은 잡음 벡터를 사용한다. 기존 파이프라인은
synth_validation_fifth.py에서 그대로 불러오며, 이 파일에서는 수정하지 않는다.

주의
  connected 정답은 전체 직사각형 입력공간에서 연결된 slab이다. 실제 레시피와
  유사하게 표본화한 관측점 자체의 support가 연속이라는 뜻은 아니다. 따라서 이
  실험은 원 방법론의 과분할 성향을 검증하며, 실제 Concrete 고성능 영역의 연결성을
  증명하지 않는다.
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

import synth_validation_fifth as original


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
DEFAULT_OUTPUT_DIR = HERE

DEFAULT_N = 1200
DEFAULT_X_SEED = 20260826
DEFAULT_NOISE_SEED = 20260827
DEFAULT_PIPELINE_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_NOISE_STD = 0.05
MIN_SUPPORT_RATIO = 0.10
HIGH_DESIRABILITY = 0.85
BACKGROUND_DESIRABILITY = 0.10

PROJECTION_FEATURES = ('Cement', 'Water', 'FineAgg')
CONNECTED_QUANTILES = ((0.275, 0.725),)
SEPARATED_QUANTILES = (
    (0.100, 0.250),
    (0.425, 0.575),
    (0.750, 0.900),
)

RUN_COLUMNS = [
    'scenario', 'pipeline_seed', 'status', 'error', 'n_samples',
    'true_components', 'truth_fraction', 'min_support', 'n_boxes',
    'n_memb_star', 'K_star', 'N_eff', 'ECC', 'rho_eff',
    'legacy_mean_f1', 'legacy_unmatched_truth', 'raw_count_match',
    'mapped_cluster_count', 'unmapped_cluster_count',
    'merged_cluster_count', 'over_split_factor', 'excess_cluster_count',
    'missing_component_count', 'split_component_count',
    'one_to_one_recovery', 'mean_assigned_cluster_purity',
    'mean_component_union_coverage', 'min_component_union_coverage',
    'mean_best_cluster_coverage',
]

COMPONENT_COLUMNS = [
    'scenario', 'pipeline_seed', 'component', 't_lower', 't_upper',
    'truth_size', 'truth_fraction', 'assigned_cluster_count',
    'union_coverage', 'best_cluster_coverage',
]

CLUSTER_COLUMNS = [
    'scenario', 'pipeline_seed', 'effective_cluster', 'hierarchy_label',
    'member_box_count', 'cluster_size', 'assigned_component',
    'overlap', 'purity', 'truth_coverage', 'f1',
    'overlapped_truth_count', 'assignment_tie',
]

SUMMARY_COLUMNS = [
    'scenario', 'true_components', 'attempted_runs', 'successful_runs',
    'failed_runs', 'mean_N_eff', 'sd_N_eff', 'raw_count_match_rate',
    'one_to_one_recovery_rate', 'mean_over_split_factor',
    'mean_excess_cluster_count', 'mean_unmapped_cluster_count',
    'mean_missing_component_count', 'mean_split_component_count',
    'mean_legacy_f1', 'mean_assigned_cluster_purity',
    'mean_component_union_coverage', 'mean_best_cluster_coverage',
]


def normalized_projection(X, real_X):
    """Cement, Water, FineAgg의 정규화 값 평균을 1차원 투영값으로 만든다."""
    indices = [original.FEAT_NAMES.index(name) for name in PROJECTION_FEATURES]
    lo = real_X.min(axis=0)
    hi = real_X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = (X - lo) / span
    return normalized[:, indices].mean(axis=1)


def intervals_from_quantiles(t, quantile_intervals):
    """분위수 구간을 실제 t 경계값으로 변환한다."""
    return [
        (float(np.quantile(t, q_low)), float(np.quantile(t, q_high)))
        for q_low, q_high in quantile_intervals
    ]


def labels_from_intervals(t, intervals):
    """구간 밖은 -1, 각 고성능 component는 0부터 시작하는 정답 label을 부여한다."""
    labels = np.full(len(t), -1, dtype=int)
    for component, (lower, upper) in enumerate(intervals):
        mask = (t >= lower) & (t <= upper)
        if np.any(labels[mask] >= 0):
            raise ValueError('정답 고성능 구간이 서로 겹칩니다.')
        labels[mask] = component
    return labels


def make_control(t, quantile_intervals, shared_noise, name):
    """정답 interval membership과 plateau desirability를 갖는 통제군을 만든다."""
    intervals = intervals_from_quantiles(t, quantile_intervals)
    labels = labels_from_intervals(t, intervals)
    means = np.where(labels >= 0, HIGH_DESIRABILITY, BACKGROUND_DESIRABILITY)
    desirability = np.clip(means + shared_noise, 0.0, 1.0)
    truth_sets = [
        set(np.where(labels == component)[0].tolist())
        for component in range(len(intervals))
    ]
    return {
        'name': name,
        'D': desirability,
        'labels': labels,
        'intervals': intervals,
        'truth_sets': truth_sets,
    }


def validate_control_design(controls, n_samples):
    """두 통제군의 정답 개수, 크기 균형, 구간 분리를 확인한다."""
    connected = controls['connected']
    separated = controls['separated']

    if len(connected['truth_sets']) != 1:
        raise AssertionError('connected 통제군의 정답 component가 1개가 아닙니다.')
    if len(separated['truth_sets']) != 3:
        raise AssertionError('separated 통제군의 정답 component가 3개가 아닙니다.')
    if any(len(truth) == 0 for control in controls.values()
           for truth in control['truth_sets']):
        raise AssertionError('관측치가 하나도 없는 정답 component가 있습니다.')

    separated_sets = separated['truth_sets']
    for i in range(len(separated_sets)):
        for j in range(i + 1, len(separated_sets)):
            if separated_sets[i] & separated_sets[j]:
                raise AssertionError('separated 정답 component가 서로 겹칩니다.')

    connected_fraction = sum(map(len, connected['truth_sets'])) / n_samples
    separated_fraction = sum(map(len, separated['truth_sets'])) / n_samples
    tolerance = max(0.05, 5.0 / n_samples)
    if abs(connected_fraction - separated_fraction) > tolerance:
        raise AssertionError(
            '두 통제군의 총 고성능 관측치 비율 차이가 너무 큽니다: '
            f'{connected_fraction:.3f} vs {separated_fraction:.3f}'
        )

    intervals = separated['intervals']
    if any(intervals[i][1] >= intervals[i + 1][0]
           for i in range(len(intervals) - 1)):
        raise AssertionError('separated 고성능 구간 사이에 gap이 없습니다.')


def generate_paired_controls(real_X, n_samples, x_seed, noise_seed, noise_std):
    """동일한 X와 잡음을 사용하는 connected/separated 통제군을 생성한다."""
    X = original.sample_recipe_replicate_inputs(real_X, n_samples, seed=x_seed)
    t = normalized_projection(X, real_X)
    shared_noise = np.random.default_rng(noise_seed).normal(
        0.0, noise_std, size=n_samples
    )
    controls = {
        'connected': make_control(
            t, CONNECTED_QUANTILES, shared_noise, name='connected'
        ),
        'separated': make_control(
            t, SEPARATED_QUANTILES, shared_noise, name='separated'
        ),
    }
    validate_control_design(controls, n_samples)
    return X, t, shared_noise, controls


def extract_effective_clusters(boxes, labels, n_memb_star):
    """effective cluster마다 멤버 박스 관측치의 합집합을 만든다."""
    clusters = []
    for hierarchy_label in np.unique(labels):
        members = np.where(labels == hierarchy_label)[0]
        if len(members) < n_memb_star:
            continue
        union = set()
        for member in members:
            union.update(boxes[member]['idx'].tolist())
        clusters.append({
            'hierarchy_label': int(hierarchy_label),
            'member_box_count': int(len(members)),
            'indices': union,
        })
    return clusters


def evaluate_cluster_topology(effective_clusters, truth_sets, scenario,
                              pipeline_seed, intervals=None, n_total=None):
    """effective cluster를 정답 component에 배정하고 과분할·복원 지표를 계산한다."""
    if n_total is None:
        all_indices = set().union(*truth_sets) if truth_sets else set()
        n_total = max(all_indices) + 1 if all_indices else 0
    if intervals is None:
        intervals = [(np.nan, np.nan)] * len(truth_sets)

    cluster_rows = []
    assigned_by_component = [[] for _ in truth_sets]
    merged_cluster_count = 0

    for cluster_index, cluster in enumerate(effective_clusters):
        cluster_set = cluster['indices']
        overlaps = np.array(
            [len(cluster_set & truth) for truth in truth_sets], dtype=int
        )
        positive_count = int(np.sum(overlaps > 0))
        merged_cluster_count += int(positive_count > 1)
        max_overlap = int(overlaps.max()) if len(overlaps) else 0

        if max_overlap > 0:
            winners = np.where(overlaps == max_overlap)[0]
            assigned = int(winners[0])
            assignment_tie = len(winners) > 1
            assigned_by_component[assigned].append(cluster_index)
            truth_size = len(truth_sets[assigned])
            purity = max_overlap / len(cluster_set) if cluster_set else 0.0
            truth_coverage = max_overlap / truth_size if truth_size else 0.0
            f1 = (
                2 * purity * truth_coverage / (purity + truth_coverage)
                if purity + truth_coverage > 0 else 0.0
            )
        else:
            assigned = -1
            assignment_tie = False
            purity = 0.0
            truth_coverage = 0.0
            f1 = 0.0

        cluster_rows.append({
            'scenario': scenario,
            'pipeline_seed': pipeline_seed,
            'effective_cluster': cluster_index,
            'hierarchy_label': cluster['hierarchy_label'],
            'member_box_count': cluster['member_box_count'],
            'cluster_size': len(cluster_set),
            'assigned_component': assigned,
            'overlap': max_overlap,
            'purity': purity,
            'truth_coverage': truth_coverage,
            'f1': f1,
            'overlapped_truth_count': positive_count,
            'assignment_tie': bool(assignment_tie),
        })

    component_rows = []
    component_cluster_counts = []
    for component, truth in enumerate(truth_sets):
        assigned_indices = assigned_by_component[component]
        component_cluster_counts.append(len(assigned_indices))
        assigned_sets = [effective_clusters[i]['indices'] for i in assigned_indices]
        assigned_union = set().union(*assigned_sets) if assigned_sets else set()
        union_coverage = len(assigned_union & truth) / len(truth) if truth else 0.0
        best_coverage = max(
            (len(cluster_set & truth) / len(truth) for cluster_set in assigned_sets),
            default=0.0,
        ) if truth else 0.0
        lower, upper = intervals[component]
        component_rows.append({
            'scenario': scenario,
            'pipeline_seed': pipeline_seed,
            'component': component,
            't_lower': lower,
            't_upper': upper,
            'truth_size': len(truth),
            'truth_fraction': len(truth) / n_total if n_total else np.nan,
            'assigned_cluster_count': len(assigned_indices),
            'union_coverage': union_coverage,
            'best_cluster_coverage': best_coverage,
        })

    counts = np.asarray(component_cluster_counts, dtype=int)
    mapped_count = int(np.sum(counts))
    unmapped_count = len(effective_clusters) - mapped_count
    purities = [
        row['purity'] for row in cluster_rows if row['assigned_component'] >= 0
    ]
    union_coverages = [row['union_coverage'] for row in component_rows]
    best_coverages = [row['best_cluster_coverage'] for row in component_rows]
    true_components = len(truth_sets)

    run_metrics = {
        'mapped_cluster_count': mapped_count,
        'unmapped_cluster_count': unmapped_count,
        'merged_cluster_count': merged_cluster_count,
        'over_split_factor': (
            mapped_count / true_components if true_components else np.nan
        ),
        'excess_cluster_count': int(np.maximum(counts - 1, 0).sum()),
        'missing_component_count': int(np.sum(counts == 0)),
        'split_component_count': int(np.sum(counts > 1)),
        'one_to_one_recovery': bool(
            true_components > 0
            and np.all(counts == 1)
            and unmapped_count == 0
        ),
        'mean_assigned_cluster_purity': (
            float(np.mean(purities)) if purities else 0.0
        ),
        'mean_component_union_coverage': (
            float(np.mean(union_coverages)) if union_coverages else 0.0
        ),
        'min_component_union_coverage': (
            float(np.min(union_coverages)) if union_coverages else 0.0
        ),
        'mean_best_cluster_coverage': (
            float(np.mean(best_coverages)) if best_coverages else 0.0
        ),
    }
    return run_metrics, component_rows, cluster_rows


def run_metric_self_check():
    """작은 정답 예제로 1:1 복원, 과분할, 누락 지표를 자체 점검한다."""
    truth_sets = [{0, 1}, {4, 5}]

    one_to_one = [
        {'hierarchy_label': 1, 'member_box_count': 5, 'indices': {0, 1}},
        {'hierarchy_label': 2, 'member_box_count': 5, 'indices': {4, 5}},
    ]
    metrics, _, _ = evaluate_cluster_topology(
        one_to_one, truth_sets, 'self-check', 0, n_total=6
    )
    assert metrics['one_to_one_recovery']
    assert metrics['excess_cluster_count'] == 0
    assert metrics['missing_component_count'] == 0

    over_split = [
        {'hierarchy_label': 1, 'member_box_count': 5, 'indices': {0}},
        {'hierarchy_label': 2, 'member_box_count': 5, 'indices': {1}},
        {'hierarchy_label': 3, 'member_box_count': 5, 'indices': {4, 5}},
    ]
    metrics, _, _ = evaluate_cluster_topology(
        over_split, truth_sets, 'self-check', 0, n_total=6
    )
    assert metrics['excess_cluster_count'] == 1
    assert metrics['split_component_count'] == 1
    assert not metrics['one_to_one_recovery']

    missing = [
        {'hierarchy_label': 1, 'member_box_count': 5, 'indices': {0, 1}},
        {'hierarchy_label': 2, 'member_box_count': 5, 'indices': {2, 3}},
    ]
    metrics, _, _ = evaluate_cluster_topology(
        missing, truth_sets, 'self-check', 0, n_total=6
    )
    assert metrics['missing_component_count'] == 1
    assert metrics['unmapped_cluster_count'] == 1


def failed_run_row(scenario, pipeline_seed, control, n_samples, min_support,
                   error, n_boxes=0):
    """예상 가능한 파이프라인 실패도 CSV에 남길 수 있도록 행을 만든다."""
    row = {column: np.nan for column in RUN_COLUMNS}
    row.update({
        'scenario': scenario,
        'pipeline_seed': pipeline_seed,
        'status': 'failed',
        'error': error,
        'n_samples': n_samples,
        'true_components': len(control['truth_sets']),
        'truth_fraction': sum(map(len, control['truth_sets'])) / n_samples,
        'min_support': min_support,
        'n_boxes': n_boxes,
    })
    return row


def run_pipeline_once(X, control, pipeline_seed, min_support):
    """기존 MRS-PRIM 파이프라인을 한 통제군·한 seed에서 그대로 실행한다."""
    scenario = control['name']
    n_samples = len(X)
    boxes = original.build_boxes(X, control['D'], min_support, pipeline_seed)
    if len(boxes) < 5:
        return (
            failed_run_row(
                scenario, pipeline_seed, control, n_samples, min_support,
                '박스 생성 실패(min_support 과도)', n_boxes=len(boxes),
            ),
            [],
            [],
        )

    Z, sets, n_boxes = original.build_linkage(boxes)
    final = original.full_grid_search(Z, sets, n_boxes)
    if final is None:
        return (
            failed_run_row(
                scenario, pipeline_seed, control, n_samples, min_support,
                'rho_Limit 만족 K 없음', n_boxes=n_boxes,
            ),
            [],
            [],
        )

    n_memb_star, K_star, _, ecc, rho = final
    labels = fcluster(Z, K_star, criterion='maxclust')
    legacy = original.score_against_truth(
        boxes, labels, n_memb_star, control['truth_sets'], n_samples
    )
    effective_clusters = extract_effective_clusters(
        boxes, labels, n_memb_star
    )
    topology, component_rows, cluster_rows = evaluate_cluster_topology(
        effective_clusters,
        control['truth_sets'],
        scenario,
        pipeline_seed,
        intervals=control['intervals'],
        n_total=n_samples,
    )

    run_row = {
        'scenario': scenario,
        'pipeline_seed': pipeline_seed,
        'status': 'ok',
        'error': '',
        'n_samples': n_samples,
        'true_components': len(control['truth_sets']),
        'truth_fraction': sum(map(len, control['truth_sets'])) / n_samples,
        'min_support': min_support,
        'n_boxes': n_boxes,
        'n_memb_star': n_memb_star,
        'K_star': K_star,
        'N_eff': len(effective_clusters),
        'ECC': ecc,
        'rho_eff': rho,
        'legacy_mean_f1': legacy['mean_f1'],
        'legacy_unmatched_truth': legacy['unmatched_truth'],
        'raw_count_match': bool(len(effective_clusters) == len(control['truth_sets'])),
        **topology,
    }
    return run_row, component_rows, cluster_rows


def summarize_runs(run_frame):
    """통제군별 반복 실행 결과를 평균·비율로 요약한다."""
    rows = []
    for scenario in ('connected', 'separated'):
        attempted = run_frame[run_frame['scenario'] == scenario]
        successful = attempted[attempted['status'] == 'ok']
        true_components = (
            int(attempted['true_components'].iloc[0]) if len(attempted) else np.nan
        )
        if len(successful):
            summary = {
                'mean_N_eff': successful['N_eff'].mean(),
                'sd_N_eff': successful['N_eff'].std(ddof=0),
                'raw_count_match_rate': successful['raw_count_match'].mean(),
                'one_to_one_recovery_rate': successful['one_to_one_recovery'].mean(),
                'mean_over_split_factor': successful['over_split_factor'].mean(),
                'mean_excess_cluster_count': successful['excess_cluster_count'].mean(),
                'mean_unmapped_cluster_count': successful['unmapped_cluster_count'].mean(),
                'mean_missing_component_count': successful['missing_component_count'].mean(),
                'mean_split_component_count': successful['split_component_count'].mean(),
                'mean_legacy_f1': successful['legacy_mean_f1'].mean(),
                'mean_assigned_cluster_purity': successful['mean_assigned_cluster_purity'].mean(),
                'mean_component_union_coverage': successful['mean_component_union_coverage'].mean(),
                'mean_best_cluster_coverage': successful['mean_best_cluster_coverage'].mean(),
            }
        else:
            summary = {
                column: np.nan for column in SUMMARY_COLUMNS
                if column not in {
                    'scenario', 'true_components', 'attempted_runs',
                    'successful_runs', 'failed_runs'
                }
            }
        rows.append({
            'scenario': scenario,
            'true_components': true_components,
            'attempted_runs': len(attempted),
            'successful_runs': len(successful),
            'failed_runs': len(attempted) - len(successful),
            **summary,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def print_control_design(t, shared_noise, controls):
    """파이프라인 실행 전, 정답 통제군이 의도대로 만들어졌는지 출력한다."""
    print('=' * 78)
    print('통제 합성실험 설계 확인')
    print('=' * 78)
    print(f'입력 관측치: {len(t)}개')
    print(f'투영값 t: 정규화한 {", ".join(PROJECTION_FEATURES)}의 평균')
    print(f'공유 noise: mean={shared_noise.mean():+.4f}, sd={shared_noise.std():.4f}')

    for scenario in ('connected', 'separated'):
        control = controls[scenario]
        total_high = sum(map(len, control['truth_sets']))
        print(f'\n[{scenario}] 정답 component={len(control["truth_sets"])}개, '
              f'고성능 관측치={total_high}/{len(t)} ({total_high/len(t):.1%})')
        for component, ((lower, upper), truth) in enumerate(
                zip(control['intervals'], control['truth_sets'])):
            print(f'  component {component}: t=[{lower:.4f}, {upper:.4f}], '
                  f'n={len(truth)}')


def print_run_result(run_row):
    """한 번의 파이프라인 결과를 짧게 출력한다."""
    if run_row['status'] != 'ok':
        print(f'  [{run_row["scenario"]}] 실패: {run_row["error"]}')
        return
    print(
        f'  [{run_row["scenario"]}] N_eff={run_row["N_eff"]}, '
        f'정답={run_row["true_components"]}, '
        f'over-split={run_row["over_split_factor"]:.2f}, '
        f'excess={run_row["excess_cluster_count"]}, '
        f'missing={run_row["missing_component_count"]}, '
        f'1:1={"O" if run_row["one_to_one_recovery"] else "X"}, '
        f'ECC={run_row["ECC"]:.3f}, F1={run_row["legacy_mean_f1"]:.3f}'
    )


def print_summary(summary):
    """connected/separated 비교와 올바른 해석 기준을 출력한다."""
    print('\n' + '=' * 78)
    print('반복 실행 요약')
    print('=' * 78)
    display_columns = [
        'scenario', 'true_components', 'successful_runs', 'mean_N_eff',
        'mean_over_split_factor', 'one_to_one_recovery_rate',
        'mean_legacy_f1', 'mean_component_union_coverage',
    ]
    print(summary[display_columns].to_string(index=False, formatters={
        'mean_N_eff': '{:.2f}'.format,
        'mean_over_split_factor': '{:.2f}'.format,
        'one_to_one_recovery_rate': '{:.1%}'.format,
        'mean_legacy_f1': '{:.3f}'.format,
        'mean_component_union_coverage': '{:.3f}'.format,
    }))
    print('\n해석 기준')
    print('- connected(정답 1개)에서 over-split factor가 반복적으로 1보다 크면 과분할 신호입니다.')
    print('- separated(정답 3개)의 1:1 복원률이 높으면 서로 다른 영역을 구분할 능력은 있다는 뜻입니다.')
    print('- 두 통제군 모두 cluster 수가 많고 불안정하면 과분할만이 아니라 전반적 cluster 불안정으로 해석합니다.')
    print('- purity와 coverage가 낮은 작은 cluster가 많다면 N_eff만으로 결론내리지 않습니다.')


def save_results(output_dir, run_rows, component_rows, cluster_rows):
    """run/component/cluster/summary 네 CSV를 저장한다."""
    os.makedirs(output_dir, exist_ok=True)
    run_frame = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    component_frame = pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)
    cluster_frame = pd.DataFrame(cluster_rows, columns=CLUSTER_COLUMNS)
    summary_frame = summarize_runs(run_frame)

    paths = {
        'runs': os.path.join(output_dir, 'connectivity_validation_runs.csv'),
        'components': os.path.join(output_dir, 'connectivity_validation_components.csv'),
        'clusters': os.path.join(output_dir, 'connectivity_validation_clusters.csv'),
        'summary': os.path.join(output_dir, 'connectivity_validation_summary.csv'),
    }
    run_frame.to_csv(paths['runs'], index=False, encoding='utf-8-sig')
    component_frame.to_csv(paths['components'], index=False, encoding='utf-8-sig')
    cluster_frame.to_csv(paths['clusters'], index=False, encoding='utf-8-sig')
    summary_frame.to_csv(paths['summary'], index=False, encoding='utf-8-sig')
    return summary_frame, paths


def parse_int_list(value):
    values = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not values:
        raise argparse.ArgumentTypeError('정수 seed를 하나 이상 입력해야 합니다.')
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description='정답 연결성 통제군으로 원 MRS-PRIM의 과분할 여부를 검증합니다.'
    )
    parser.add_argument('--csv', default=CSV_PATH)
    parser.add_argument('--n', type=int, default=DEFAULT_N)
    parser.add_argument('--x-seed', type=int, default=DEFAULT_X_SEED)
    parser.add_argument('--noise-seed', type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument(
        '--pipeline-seeds', type=parse_int_list,
        default=DEFAULT_PIPELINE_SEEDS,
        help='쉼표로 구분한 MRS-PRIM seed (기본: 0,1,2,3,4)',
    )
    parser.add_argument('--noise-std', type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--design-only', action='store_true',
        help='통제군 설계만 확인하고 MRS-PRIM은 실행하지 않습니다.',
    )
    return parser.parse_args()


def validate_args(args):
    if args.n < 100:
        raise ValueError('--n은 100 이상이어야 합니다.')
    if args.noise_std < 0:
        raise ValueError('--noise-std는 0 이상이어야 합니다.')
    if not args.pipeline_seeds:
        raise ValueError('--pipeline-seeds를 하나 이상 입력해야 합니다.')


def main():
    args = parse_args()
    validate_args(args)
    run_metric_self_check()

    real_X = original.load_real_X(args.csv)
    X, t, shared_noise, controls = generate_paired_controls(
        real_X,
        n_samples=args.n,
        x_seed=args.x_seed,
        noise_seed=args.noise_seed,
        noise_std=args.noise_std,
    )
    print_control_design(t, shared_noise, controls)

    if args.design_only:
        print('\n설계 검증 완료: MRS-PRIM은 실행하지 않았습니다.')
        return

    min_support = max(20, int(args.n * MIN_SUPPORT_RATIO))
    run_rows = []
    component_rows = []
    cluster_rows = []

    print('\n' + '=' * 78)
    print(f'원 MRS-PRIM 실행 (min_support={min_support})')
    print('=' * 78)
    for pipeline_seed in args.pipeline_seeds:
        print(f'\npipeline seed={pipeline_seed}')
        for scenario in ('connected', 'separated'):
            run_row, run_components, run_clusters = run_pipeline_once(
                X, controls[scenario], pipeline_seed, min_support
            )
            run_rows.append(run_row)
            component_rows.extend(run_components)
            cluster_rows.extend(run_clusters)
            print_run_result(run_row)

    summary, paths = save_results(
        args.output_dir, run_rows, component_rows, cluster_rows
    )
    print_summary(summary)
    print('\n결과 파일')
    for path in paths.values():
        print(f'- {path}')


if __name__ == '__main__':
    main()
