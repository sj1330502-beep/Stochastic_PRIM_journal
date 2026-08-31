"""
Naval 데이터에서 원 논문 방식과 validation-peak 방식을 공정하게 비교한다.

원 논문
    rho_eff >= rho_limit인 후보 중 ECC 최대

제안 방식
    score = D_valid의 보수적 하한
            + beta * log(validation coverage)
            - lambda * (K / M)

데이터 사용 순서
    inner training  : MRS-PRIM 박스와 클러스터 생성
    inner validation: 두 방식의 (N_Memb, K) 선택
    outer training  : 선택된 설정으로 최종 모델 재구축
    confirmation    : 두 방식의 최종 성능을 한 번만 비교

실행 예시
    python idea6/idea6_naval_original_vs_validation_peak.py
    python idea6/idea6_naval_original_vs_validation_peak.py --beta 0.05 --lambda-k 0.10 --min-valid 50
"""
import argparse
from pathlib import Path

import numpy as np

from idea6_naval_original_vs_method4 import (
    CSV_PATH,
    RESPONSE_COLS,
    build_boxes,
    build_linkage,
    build_shared_candidates,
    desirability_from_bounds,
    fit_desirability_bounds,
    load_naval,
    make_k_grid,
    precompute_raw_clusters,
    select_original,
)


OUTER_TRAIN_RATIO = 0.70
INNER_TRAIN_RATIO = 0.70
OUTER_SPLIT_SEED = 0
INNER_SPLIT_SEED = 1
RHO_LIMIT = 0.60


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("0 이상의 값을 입력해야 합니다.")
    return value


def ratio(value):
    value = float(value)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("0보다 크고 1보다 작은 값을 입력해야 합니다.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="원 논문 방식과 validation-peak 방식을 Naval 데이터에서 비교합니다."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument(
        "--beta", type=nonnegative_float, default=0.05,
        help="validation coverage 보상 강도 (기본값: 0.05)",
    )
    parser.add_argument(
        "--lambda-k", type=nonnegative_float, default=0.10,
        help="K/M 복잡도 벌점 강도 (기본값: 0.10)",
    )
    parser.add_argument(
        "--z", type=nonnegative_float, default=1.96,
        help="D_valid 불확실성 보정값 (기본값: 1.96; 0이면 평균만 사용)",
    )
    parser.add_argument(
        "--min-valid", type=int, default=50,
        help="후보 범위에 필요한 최소 validation 관측치 수 (기본값: 50)",
    )
    parser.add_argument(
        "--rho-limit", type=ratio, default=RHO_LIMIT,
        help="원 논문 방식의 최소 rho_eff (기본값: 0.60)",
    )
    parser.add_argument(
        "--outer-train-ratio", type=ratio, default=OUTER_TRAIN_RATIO,
        help="전체 데이터 중 outer training 비율 (기본값: 0.70)",
    )
    parser.add_argument(
        "--inner-train-ratio", type=ratio, default=INNER_TRAIN_RATIO,
        help="outer training 중 inner training 비율 (기본값: 0.70)",
    )
    parser.add_argument("--outer-seed", type=int, default=OUTER_SPLIT_SEED)
    parser.add_argument("--inner-seed", type=int, default=INNER_SPLIT_SEED)
    args = parser.parse_args()
    if args.min_valid < 2:
        parser.error("--min-valid는 2 이상이어야 합니다.")
    return args


def split_indices(indices, train_ratio, seed):
    indices = np.asarray(indices, dtype=int)
    permutation = np.random.default_rng(seed).permutation(len(indices))
    n_train = int(len(indices) * train_ratio)
    if n_train == 0 or n_train == len(indices):
        raise ValueError("두 데이터 구간에 각각 관측치가 필요합니다.")
    return indices[permutation[:n_train]], indices[permutation[n_train:]]


def make_candidates(boxes, n_observations):
    hierarchy, observation_sets = build_linkage(boxes, n_observations)
    k_grid = make_k_grid(len(boxes))
    raw_by_k = precompute_raw_clusters(
        hierarchy, observation_sets, boxes, k_grid
    )
    candidates = build_shared_candidates(raw_by_k, len(boxes), k_grid)
    return candidates, k_grid


def region_values(boxes, candidate, X_train, X_eval, D_eval):
    member_ids = candidate["largest_members"]
    union_idx = np.unique(
        np.concatenate([boxes[i]["idx"] for i in member_ids])
    )
    lower = X_train[union_idx].min(axis=0)
    upper = X_train[union_idx].max(axis=0)
    mask = np.all((X_eval >= lower) & (X_eval <= upper), axis=1)
    return D_eval[mask], lower, upper


def validation_metrics(
    boxes,
    candidate,
    X_train,
    X_valid,
    D_valid,
    beta,
    lambda_k,
    z,
):
    values, _, _ = region_values(
        boxes, candidate, X_train, X_valid, D_valid
    )
    n_valid = len(values)
    coverage = n_valid / len(X_valid)
    if n_valid == 0:
        return None

    mean_d = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else np.inf
    )
    conservative_d = mean_d - z * standard_error
    score = (
        conservative_d
        + beta * np.log(coverage)
        - lambda_k * (candidate["K"] / len(boxes))
    )
    return {
        "score": score,
        "mean_d": mean_d,
        "standard_error": standard_error,
        "conservative_d": conservative_d,
        "n_valid": n_valid,
        "coverage": coverage,
    }


def select_validation_peak(
    candidates,
    boxes,
    X_train,
    X_valid,
    D_valid,
    beta,
    lambda_k,
    z,
    min_valid,
):
    feasible = []
    for candidate in candidates:
        metrics = validation_metrics(
            boxes, candidate, X_train, X_valid, D_valid,
            beta, lambda_k, z,
        )
        if metrics is None or metrics["n_valid"] < min_valid:
            continue
        feasible.append({**candidate, "validation": metrics})

    if not feasible:
        raise RuntimeError(
            f"n_valid >= {min_valid}를 만족하는 후보가 없습니다. "
            "--min-valid를 낮춰 보세요."
        )

    # 점수가 같으면 더 작은 K, 그다음 더 작은 N_Memb를 선택한다.
    return max(
        feasible,
        key=lambda c: (
            c["validation"]["score"],
            -c["K"],
            -c["n_memb"],
        ),
    )


def find_candidate(candidates, selected):
    for candidate in candidates:
        if (
            candidate["n_memb"] == selected["n_memb"]
            and candidate["K"] == selected["K"]
        ):
            return candidate
    raise RuntimeError("outer training에서 선택된 (N_Memb, K)를 찾지 못했습니다.")


def evaluate_final(boxes, candidate, X_train, X_confirm, D_confirm):
    values, lower, upper = region_values(
        boxes, candidate, X_train, X_confirm, D_confirm
    )
    n_confirm = len(values)
    return {
        "d_confirm": float(values.mean()) if n_confirm else float("nan"),
        "n_confirm": n_confirm,
        "coverage": n_confirm / len(X_confirm),
        "lower": lower,
        "upper": upper,
    }


def format_number(value, digits=4):
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def print_inner_selection(name, candidate, metrics):
    print(
        f"  {name:<18} N_Memb*={candidate['n_memb']}, K*={candidate['K']}, "
        f"ECC={candidate['ecc']:.4f}, rho={candidate['rho_eff']:.3f}, "
        f"D_valid={format_number(metrics['mean_d'])}, "
        f"n_valid={metrics['n_valid']}, coverage={metrics['coverage']:.3%}"
    )
    if "score" in metrics:
        print(
            f"  {'':18} 보수적 D={metrics['conservative_d']:.4f}, "
            f"최종 선택 점수={metrics['score']:.4f}"
        )


def main():
    args = parse_args()
    df, X_all, feature_names = load_naval(args.csv)

    outer_train_idx, confirm_idx = split_indices(
        np.arange(len(df)), args.outer_train_ratio, args.outer_seed
    )
    inner_train_idx, valid_idx = split_indices(
        outer_train_idx, args.inner_train_ratio, args.inner_seed
    )

    print("=" * 105)
    print("Naval: 원 논문 방식 vs validation-peak 방식")
    print("=" * 105)
    print(
        f"전체 {len(df)} | inner training {len(inner_train_idx)} | "
        f"inner validation {len(valid_idx)} | confirmation {len(confirm_idx)}"
    )
    print(
        f"제안 점수: 보수적 D + {args.beta:g}*log(coverage) "
        f"- {args.lambda_k:g}*(K/M), z={args.z:g}, min_valid={args.min_valid}"
    )

    # ---------- inner validation에서 두 방식의 설정 선택
    inner_bounds = fit_desirability_bounds(df, inner_train_idx)
    X_inner = X_all[inner_train_idx]
    X_valid = X_all[valid_idx]
    D_inner = desirability_from_bounds(df, inner_train_idx, inner_bounds)
    D_valid = desirability_from_bounds(df, valid_idx, inner_bounds)

    print("\n[1] inner training으로 공통 MRS-PRIM 박스 생성")
    inner_boxes = build_boxes(X_inner, D_inner)
    inner_candidates, inner_k_grid = make_candidates(inner_boxes, len(X_inner))
    print(
        f"    박스 M={len(inner_boxes)}, K={inner_k_grid[0]}~{inner_k_grid[-1]}, "
        f"후보 {len(inner_candidates)}개"
    )

    original_inner = select_original(inner_candidates, args.rho_limit)
    proposed_inner = select_validation_peak(
        inner_candidates,
        inner_boxes,
        X_inner,
        X_valid,
        D_valid,
        args.beta,
        args.lambda_k,
        args.z,
        args.min_valid,
    )

    original_valid = validation_metrics(
        inner_boxes, original_inner, X_inner, X_valid, D_valid,
        args.beta, args.lambda_k, args.z,
    )

    print("\n[2] inner validation 선택 결과")
    print_inner_selection("원 논문", original_inner, original_valid)
    print_inner_selection(
        "제안 방식", proposed_inner, proposed_inner["validation"]
    )

    # ---------- outer training 전체로 재구축하고 고정 설정 적용
    outer_bounds = fit_desirability_bounds(df, outer_train_idx)
    X_outer = X_all[outer_train_idx]
    X_confirm = X_all[confirm_idx]
    D_outer = desirability_from_bounds(df, outer_train_idx, outer_bounds)
    D_confirm = desirability_from_bounds(df, confirm_idx, outer_bounds)

    print("\n[3] outer training 전체로 공통 파이프라인 재구축")
    outer_boxes = build_boxes(X_outer, D_outer)
    outer_candidates, outer_k_grid = make_candidates(outer_boxes, len(X_outer))
    print(
        f"    박스 M={len(outer_boxes)}, K={outer_k_grid[0]}~{outer_k_grid[-1]}, "
        f"후보 {len(outer_candidates)}개"
    )

    original_final = find_candidate(outer_candidates, original_inner)
    proposed_final = find_candidate(outer_candidates, proposed_inner)
    final_rows = []
    for name, candidate in (
        ("원 논문", original_final),
        ("제안 방식", proposed_final),
    ):
        final_rows.append((
            name,
            candidate,
            evaluate_final(
                outer_boxes, candidate, X_outer, X_confirm, D_confirm
            ),
        ))

    print("\n[4] untouched confirmation 최종 비교")
    print("=" * 105)
    print(
        f'{"방식":<14} {"N_Memb*":>8} {"K*":>6} {"N_eff":>7} '
        f'{"ECC":>8} {"rho":>8} {"n_conf":>8} {"coverage":>10} {"D_conf":>10}'
    )
    for name, candidate, result in final_rows:
        print(
            f'{name:<14} {candidate["n_memb"]:>8} {candidate["K"]:>6} '
            f'{candidate["n_eff"]:>7} {candidate["ecc"]:>8.4f} '
            f'{candidate["rho_eff"]:>8.3f} {result["n_confirm"]:>8} '
            f'{result["coverage"]:>10.3%} '
            f'{format_number(result["d_confirm"]):>10}'
        )

    original_d = final_rows[0][2]["d_confirm"]
    proposed_d = final_rows[1][2]["d_confirm"]
    print("=" * 105)
    if np.isfinite(original_d) and np.isfinite(proposed_d):
        print(f"D_conf 차이(제안 - 원 논문): {proposed_d - original_d:+.4f}")
    else:
        print("한 방식의 confirmation 범위가 비어 D_conf 차이를 계산하지 못했습니다.")

    print("\n선택된 제안 방식의 최종 운전 범위")
    proposed_result = final_rows[1][2]
    for name, lower, upper in zip(
        feature_names, proposed_result["lower"], proposed_result["upper"]
    ):
        print(f"  {name:>38}: [{lower:>12.5f}, {upper:>12.5f}]")

    print(
        "\n주의: beta/lambda/z를 이 confirmation 결과를 보고 반복 조정하면 "
        "confirmation도 학습에 사용한 셈이 됩니다. 여러 설정을 비교하려면 "
        "inner validation 결과만 보고 정한 뒤 새로운 outer seed로 확인하세요."
    )


if __name__ == "__main__":
    main()
