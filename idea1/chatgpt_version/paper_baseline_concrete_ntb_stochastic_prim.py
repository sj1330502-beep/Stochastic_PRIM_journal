"""
paper_baseline_concrete_ntb_stochastic_prim.py

Baseline implementation for the granularity-based clustering framework,
adapted to the Concrete Compressive Strength dataset with a single-response
NTB desirability objective.

Only the response-optimization front end is changed:
    Concrete strength y
      -> NTB desirability d(y), target = 60 MPa
      -> D_i = d_i because there is only one response
      -> stochastic PRIM maximizes mean desirability D_bar inside each box

The downstream logic follows the granularity paper as closely as possible:
    stochastic PRIM subregions
      -> empirical Jaccard distance on realized observations
      -> average-linkage hierarchical clustering
      -> effective clusters
      -> cluster compactness = all-member intersection / all-member union
      -> ECC = unweighted mean compactness of effective clusters
      -> maximize ECC subject to rho_eff >= 0.60

Concrete-specific adaptation of the random-subspace scope:
- The granularity case study used P=13 and S={7,8,9,10}, T=500 each (2000 boxes).
- The concrete dataset has P=8, so the default here uses four consecutive
  stochastic subspace sizes below P: S={4,5,6,7}, T=500 each (2000 boxes).
- This preserves the original case study's 4 subspace settings, 500 trials per
  setting, and total of 2000 subregions while respecting S < P.
- These S values are an adaptation, not values explicitly reported in the paper.

Reproducibility note:
The papers do not report the original random seed. A fixed seed is therefore
provided only to make this implementation reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import argparse
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    csv_path: str = "concrete_dataset.csv"
    output_dir: str = "baseline_ntb_stochastic_prim_output"

    # User-specified single-response NTB desirability.
    target: float = 60.0

    # Granularity-paper PRIM settings retained where directly transferable.
    peel_alpha: float = 0.05
    min_support: int = 100

    # Concrete adaptation: P=8, so S must be < 8.
    # Four settings x 500 trials = 2000 discovered subregions,
    # matching the granularity case study's total trial count.
    subset_sizes: tuple[int, ...] = (4, 5, 6, 7)
    trials_per_subset: int = 500

    # Granularity heuristic from the original paper.
    n_memb_options: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
    rho_limit: float = 0.60
    k_grid_start: int = 100
    k_grid_stop: int = 1000
    k_grid_step: int = 50

    # Implementation-only reproducibility setting.
    random_seed: int = 2026


@dataclass
class Box:
    trial_id: int
    subset_size: int
    selected_step: int
    bounds: np.ndarray              # shape (P, 2)
    observation_indices: np.ndarray # indices in the full dataset
    mean_desirability: float
    mean_response: float
    min_response: float
    max_response: float


# ---------------------------------------------------------------------
# 1. DATA + SINGLE-RESPONSE NTB DESIRABILITY
# ---------------------------------------------------------------------

def load_concrete(csv_path: str):
    df = pd.read_csv(csv_path)

    if df.shape[1] != 9:
        raise ValueError(
            "Expected 8 input columns + 1 response column for the concrete "
            f"dataset, but found {df.shape[1]} columns."
        )

    input_cols = list(df.columns[:8])
    response_col = df.columns[8]

    X = df[input_cols].to_numpy(dtype=float)
    y = df[response_col].to_numpy(dtype=float)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Missing values were detected in the dataset.")

    return df, input_cols, response_col, X, y


def make_desirability(y_all: np.ndarray, target: float = 60.0) -> np.ndarray:
    """
    User-specified NTB desirability:

        d(y) = (y - lower) / (target - lower),   y <= target
             = (upper - y) / (upper - target),  y > target

    where lower=min(y_all), upper=max(y_all).

    Hence:
      d(lower)=0, d(target)=1, d(upper)=0.

    With a single response, the overall desirability is simply D_i = d_i.
    """
    y = np.asarray(y_all, dtype=float)
    lower = float(np.min(y))
    upper = float(np.max(y))

    if not (lower < target < upper):
        raise ValueError(
            f"NTB target must lie strictly inside the observed response range; "
            f"got lower={lower}, target={target}, upper={upper}."
        )

    d = np.empty_like(y, dtype=float)
    m1 = y <= target
    m2 = ~m1

    d[m1] = (y[m1] - lower) / (target - lower)
    d[m2] = (upper - y[m2]) / (upper - target)

    # Numerical protection only; the formula itself already targets [0,1].
    return np.clip(d, 0.0, 1.0)


# ---------------------------------------------------------------------
# 2. SINGLE-RESPONSE STOCHASTIC PRIM
# ---------------------------------------------------------------------

def mask_from_bounds(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Realized observations lying inside an axis-aligned PRIM box."""
    return np.all((X >= bounds[:, 0]) & (X <= bounds[:, 1]), axis=1)


def _box_stats(
    y: np.ndarray,
    desirability: np.ndarray,
    mask: np.ndarray,
):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    yy = y[idx]
    dd = desirability[idx]
    return (
        float(np.mean(dd)),
        float(np.mean(yy)),
        float(np.min(yy)),
        float(np.max(yy)),
    )


def _candidate_peel(
    X: np.ndarray,
    desirability: np.ndarray,
    current_mask: np.ndarray,
    current_bounds: np.ndarray,
    feature: int,
    side: str,
    peel_alpha: float,
):
    """
    PRIM percentile peel following the paper's interval construction:

      lower-tail peel: [Pr_(100*alpha), current UB]
      upper-tail peel: [current LB, Pr_(100*(1-alpha))]

    The objective is the mean desirability of realized observations retained
    by the candidate box.
    """
    current_values = X[current_mask, feature]
    if current_values.size < 2:
        return None

    if side == "lower":
        q = float(np.quantile(current_values, peel_alpha))
        candidate_mask = current_mask & (X[:, feature] >= q)
    elif side == "upper":
        q = float(np.quantile(current_values, 1.0 - peel_alpha))
        candidate_mask = current_mask & (X[:, feature] <= q)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    # Ties may cause a percentile boundary to remove no realized rows.
    # Such a candidate does not advance the peeling trajectory.
    candidate_n = int(candidate_mask.sum())
    current_n = int(current_mask.sum())
    if candidate_n == 0 or candidate_n >= current_n:
        return None

    candidate_bounds = current_bounds.copy()
    if side == "lower":
        candidate_bounds[feature, 0] = max(candidate_bounds[feature, 0], q)
    else:
        candidate_bounds[feature, 1] = min(candidate_bounds[feature, 1], q)

    objective = float(np.mean(desirability[candidate_mask]))

    return {
        "objective": objective,
        "mask": candidate_mask,
        "bounds": candidate_bounds,
        "feature": int(feature),
        "side": side,
        "threshold": q,
    }


def stochastic_prim_trial(
    X: np.ndarray,
    y: np.ndarray,
    desirability: np.ndarray,
    subset_size: int,
    trial_id: int,
    config: Config,
    rng: np.random.Generator,
) -> Box:
    """
    One stochastic PRIM trial.

    At every peeling step:
      1) sample S input features without replacement;
      2) create lower/upper candidates for those S features (2S candidates);
      3) choose the candidate with the largest mean desirability;
      4) stop when the locally selected candidate first has size < min_support.

    The best box in the valid trajectory B0,...,B_(m-1) is then selected by
    maximum mean desirability, consistent with the PRIM trajectory rule.
    """
    n, p = X.shape
    if not (1 <= subset_size < p):
        raise ValueError(
            f"Stochastic PRIM requires 1 <= S < P; got S={subset_size}, P={p}."
        )

    current_mask = np.ones(n, dtype=bool)
    current_bounds = np.column_stack([X.min(axis=0), X.max(axis=0)]).astype(float)

    # Keep the entire valid trajectory because the paper selects the box
    # having the best objective value within a trial trajectory.
    trajectory: list[tuple[int, np.ndarray, np.ndarray, float]] = []

    objective0 = float(np.mean(desirability[current_mask]))
    trajectory.append((0, current_mask.copy(), current_bounds.copy(), objective0))

    step = 0
    while True:
        step += 1

        # Core stochastic mechanism: independently resample S variables
        # at every peeling step, without replacement.
        sampled_features = rng.choice(p, size=subset_size, replace=False)

        candidates = []
        for feature in sampled_features:
            lower = _candidate_peel(
                X=X,
                desirability=desirability,
                current_mask=current_mask,
                current_bounds=current_bounds,
                feature=int(feature),
                side="lower",
                peel_alpha=config.peel_alpha,
            )
            upper = _candidate_peel(
                X=X,
                desirability=desirability,
                current_mask=current_mask,
                current_bounds=current_bounds,
                feature=int(feature),
                side="upper",
                peel_alpha=config.peel_alpha,
            )
            if lower is not None:
                candidates.append(lower)
            if upper is not None:
                candidates.append(upper)

        if not candidates:
            break

        # Local heuristic decision: choose only the best candidate at this step.
        # np.argmax is stable, so an exact tie keeps the first candidate.
        best_pos = int(np.argmax([c["objective"] for c in candidates]))
        best_candidate = candidates[best_pos]

        next_mask = best_candidate["mask"]
        next_bounds = best_candidate["bounds"]

        # Paper-style stopping rule: the first selected B_m falling below
        # the minimum support ends the trajectory; B_m itself is not retained.
        if int(next_mask.sum()) < config.min_support:
            break

        current_mask = next_mask
        current_bounds = next_bounds
        trajectory.append(
            (
                step,
                current_mask.copy(),
                current_bounds.copy(),
                float(best_candidate["objective"]),
            )
        )

    # Best objective box within the valid trajectory.
    # Stable argmax leaves exact ties at the earlier trajectory box rather than
    # injecting an additional unreported tie-breaking heuristic.
    best_idx = int(np.argmax([row[3] for row in trajectory]))
    selected_step, selected_mask, selected_bounds, selected_obj = trajectory[best_idx]

    mean_d, mean_y, min_y, max_y = _box_stats(y, desirability, selected_mask)

    return Box(
        trial_id=trial_id,
        subset_size=subset_size,
        selected_step=int(selected_step),
        bounds=selected_bounds.copy(),
        observation_indices=np.flatnonzero(selected_mask),
        mean_desirability=float(mean_d),
        mean_response=float(mean_y),
        min_response=float(min_y),
        max_response=float(max_y),
    )


def generate_stochastic_prim_boxes(
    X: np.ndarray,
    y: np.ndarray,
    desirability: np.ndarray,
    config: Config,
) -> list[Box]:
    """Generate one selected subregion from every stochastic PRIM trial."""
    rng = np.random.default_rng(config.random_seed)

    boxes: list[Box] = []
    trial_id = 0

    for S in config.subset_sizes:
        if S >= X.shape[1]:
            raise ValueError(
                f"subset_sizes contains S={S}, but P={X.shape[1]}; stochastic "
                "subspace size must satisfy S < P."
            )

        for _ in range(config.trials_per_subset):
            trial_id += 1
            boxes.append(
                stochastic_prim_trial(
                    X=X,
                    y=y,
                    desirability=desirability,
                    subset_size=S,
                    trial_id=trial_id,
                    config=config,
                    rng=rng,
                )
            )

    return boxes


# ---------------------------------------------------------------------
# 3. EMPIRICAL SIMILARITY / JACCARD DISTANCE
# ---------------------------------------------------------------------

def membership_matrix_from_boxes(
    boxes: Sequence[Box],
    X_reference: np.ndarray,
) -> np.ndarray:
    """
    M x N Boolean matrix.

    Each subregion is represented by the realized observations from one common
    reference dataset that fall within its min-max input bounds.
    """
    return np.vstack(
        [mask_from_bounds(X_reference, b.bounds) for b in boxes]
    ).astype(bool)


def empirical_jaccard_distance(membership: np.ndarray) -> np.ndarray:
    """
    Similarity(A,B) = |A intersection B| / |A union B|
    Distance(A,B)   = 1 - Similarity(A,B)
    """
    A = membership.astype(np.int32)
    intersection = A @ A.T
    sizes = A.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection

    similarity = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=float),
        where=union > 0,
    )
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


# ---------------------------------------------------------------------
# 4. HIERARCHICAL CLUSTERING
# ---------------------------------------------------------------------

def fit_average_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    condensed = squareform(distance_matrix, checks=False)
    return linkage(condensed, method="average")


def exact_k_labels(Z: np.ndarray, k: int) -> np.ndarray:
    """Cut the already-fixed dendrogram so that exactly K clusters remain."""
    return cut_tree(Z, n_clusters=[k]).reshape(-1).astype(int)


# ---------------------------------------------------------------------
# 5. EFFECTIVE CLUSTERS + ECC
# ---------------------------------------------------------------------

def cluster_compactness(member_rows: np.ndarray) -> float:
    """
    Compactness of one effective cluster:

        | intersection of observations shared by ALL member subregions |
        ----------------------------------------------------------------
        | union of observations appearing in ANY member subregion |
    """
    if member_rows.ndim != 2 or member_rows.shape[0] == 0:
        raise ValueError("member_rows must contain at least one subregion.")

    intersection_n = int(np.all(member_rows, axis=0).sum())
    union_n = int(np.any(member_rows, axis=0).sum())

    return float(intersection_n / union_n) if union_n > 0 else 0.0


def evaluate_partition(
    labels: np.ndarray,
    membership: np.ndarray,
    n_memb: int,
):
    unique_labels, counts = np.unique(labels, return_counts=True)
    effective_labels = unique_labels[counts >= n_memb]

    if effective_labels.size == 0:
        return {
            "K_actual": int(unique_labels.size),
            "N_eff": 0,
            "N_eff_members": 0,
            "rho_eff": 0.0,
            "ECC": np.nan,
            "ECC_min": np.nan,
            "ECC_max": np.nan,
        }

    compactness_values = []
    effective_member_count = 0

    for lab in effective_labels:
        member_idx = np.flatnonzero(labels == lab)
        effective_member_count += int(member_idx.size)
        compactness_values.append(
            cluster_compactness(membership[member_idx])
        )

    compactness_values = np.asarray(compactness_values, dtype=float)

    return {
        "K_actual": int(unique_labels.size),
        "N_eff": int(effective_labels.size),
        "N_eff_members": int(effective_member_count),
        "rho_eff": float(effective_member_count / membership.shape[0]),
        # Original-paper ECC is an unweighted average over effective clusters.
        "ECC": float(np.mean(compactness_values)),
        "ECC_min": float(np.min(compactness_values)),
        "ECC_max": float(np.max(compactness_values)),
    }


def build_k_grid(M: int, config: Config) -> list[int]:
    stop = min(config.k_grid_stop, M)
    grid = list(range(config.k_grid_start, stop + 1, config.k_grid_step))
    return grid if grid else [M]


def run_granularity_heuristic(
    Z: np.ndarray,
    membership: np.ndarray,
    config: Config,
):
    """
    Original-paper optimization:

        maximize ECC(K)
        subject to rho_eff(K) >= rho_limit

    independently for each N_Memb option.
    """
    M = membership.shape[0]
    k_grid = build_k_grid(M, config)

    rows = []
    labels_cache: dict[int, np.ndarray] = {}

    # The hierarchy is built once. K only changes the dendrogram cut.
    for K in k_grid:
        labels = exact_k_labels(Z, K)
        labels_cache[K] = labels

        for n_memb in config.n_memb_options:
            metrics = evaluate_partition(labels, membership, n_memb)
            rows.append(
                {
                    "N_Memb": n_memb,
                    "K": K,
                    **metrics,
                    "feasible": metrics["rho_eff"] >= config.rho_limit,
                }
            )

    trajectory = pd.DataFrame(rows)

    selected_rows = []
    selected_labels: dict[int, np.ndarray] = {}

    for n_memb in config.n_memb_options:
        feasible = trajectory[
            (trajectory["N_Memb"] == n_memb)
            & trajectory["feasible"]
            & trajectory["ECC"].notna()
        ].copy()

        if feasible.empty:
            selected_rows.append(
                {
                    "N_Memb": n_memb,
                    "K_star": np.nan,
                    "N_eff": np.nan,
                    "rho_eff": np.nan,
                    "ECC": np.nan,
                    "ECC_min": np.nan,
                    "ECC_max": np.nan,
                }
            )
            continue

        best_ecc = float(feasible["ECC"].max())
        tied = feasible[np.isclose(feasible["ECC"], best_ecc, atol=1e-12, rtol=0)]

        # If ECC is exactly tied, use the rightmost/larger K candidate.
        best = tied.sort_values("K").iloc[-1]
        K_star = int(best["K"])
        selected_labels[n_memb] = labels_cache[K_star]

        selected_rows.append(
            {
                "N_Memb": n_memb,
                "K_star": K_star,
                "N_eff": int(best["N_eff"]),
                "rho_eff": float(best["rho_eff"]),
                "ECC": float(best["ECC"]),
                "ECC_min": float(best["ECC_min"]),
                "ECC_max": float(best["ECC_max"]),
            }
        )

    return trajectory, pd.DataFrame(selected_rows), selected_labels


# ---------------------------------------------------------------------
# 6. OUTPUT HELPERS
# ---------------------------------------------------------------------

def boxes_to_dataframe(
    boxes: Sequence[Box],
    input_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []

    for b in boxes:
        row = {
            "trial_id": b.trial_id,
            "subset_size": b.subset_size,
            "selected_step": b.selected_step,
            "n_observations": int(len(b.observation_indices)),
            "mean_desirability": b.mean_desirability,
            "mean_response": b.mean_response,
            "min_response": b.min_response,
            "max_response": b.max_response,
        }

        for j, col in enumerate(input_cols):
            row[f"X{j+1}_name"] = col
            row[f"X{j+1}_lower"] = float(b.bounds[j, 0])
            row[f"X{j+1}_upper"] = float(b.bounds[j, 1])

        rows.append(row)

    return pd.DataFrame(rows)


def effective_cluster_summary(
    labels: np.ndarray,
    membership: np.ndarray,
    boxes: Sequence[Box],
    n_memb: int,
) -> pd.DataFrame:
    unique_labels, counts = np.unique(labels, return_counts=True)
    effective_labels = unique_labels[counts >= n_memb]

    rows = []
    for lab in effective_labels:
        idx = np.flatnonzero(labels == lab)
        box_d = np.asarray([boxes[i].mean_desirability for i in idx], dtype=float)
        box_n = np.asarray([len(boxes[i].observation_indices) for i in idx], dtype=int)

        rows.append(
            {
                "cluster_id": int(lab),
                "member_subregions": int(idx.size),
                "compactness": cluster_compactness(membership[idx]),
                "mean_box_desirability": float(np.mean(box_d)),
                "min_box_desirability": float(np.min(box_d)),
                "max_box_desirability": float(np.max(box_d)),
                "mean_box_size": float(np.mean(box_n)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "member_subregions",
                "compactness",
                "mean_box_desirability",
                "min_box_desirability",
                "max_box_desirability",
                "mean_box_size",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["member_subregions", "compactness"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def plot_dendrogram(Z: np.ndarray, outpath: Path):
    plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True)
    plt.xlabel("Subregion options")
    plt.ylabel("Empirical Jaccard distance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_ecc_trajectories(
    trajectory: pd.DataFrame,
    config: Config,
    outpath: Path,
):
    plt.figure(figsize=(9, 6))

    for n_memb in config.n_memb_options:
        sub = trajectory[
            (trajectory["N_Memb"] == n_memb)
            & trajectory["feasible"]
        ].sort_values("K")

        if not sub.empty:
            plt.plot(
                sub["K"],
                sub["ECC"],
                marker="o",
                label=f"N_Memb={n_memb}",
            )

    plt.xlabel("K (Total Number of Clusters)")
    plt.ylabel("ECC (Effective Cluster Compactness)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------

def run_full_pipeline(config: Config):
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, input_cols, response_col, X, y = load_concrete(config.csv_path)
    D = make_desirability(y, target=config.target)

    # Save the exact observation-level objective used by stochastic PRIM.
    desirability_df = df.copy()
    desirability_df["NTB_desirability_D"] = D
    desirability_df.to_csv(outdir / "01_data_with_ntb_desirability.csv", index=False)

    boxes = generate_stochastic_prim_boxes(X, y, D, config)
    boxes_df = boxes_to_dataframe(boxes, input_cols)
    boxes_df.to_csv(outdir / "02_stochastic_prim_subregions.csv", index=False)

    # Do not deduplicate repeatedly rediscovered subregions: rediscovery
    # frequency is part of the downstream effective-cluster evidence.
    membership = membership_matrix_from_boxes(boxes, X)
    np.save(outdir / "03_membership_matrix.npy", membership)

    distance = empirical_jaccard_distance(membership)
    np.save(outdir / "04_empirical_jaccard_distance.npy", distance)

    Z = fit_average_linkage(distance)
    np.save(outdir / "05_average_linkage_Z.npy", Z)

    trajectory, selected, selected_labels = run_granularity_heuristic(
        Z=Z,
        membership=membership,
        config=config,
    )

    trajectory.to_csv(
        outdir / "06_ecc_trajectory_all_candidates.csv",
        index=False,
    )
    selected.to_csv(
        outdir / "07_selected_macrostructure_table.csv",
        index=False,
    )

    plot_dendrogram(Z, outdir / "08_dendrogram.png")
    plot_ecc_trajectories(
        trajectory,
        config,
        outdir / "09_ecc_trajectories.png",
    )

    for n_memb, labels in selected_labels.items():
        np.save(
            outdir / f"labels_NMemb_{n_memb}.npy",
            labels,
        )
        detail = effective_cluster_summary(
            labels=labels,
            membership=membership,
            boxes=boxes,
            n_memb=n_memb,
        )
        detail.to_csv(
            outdir / f"effective_clusters_NMemb_{n_memb}.csv",
            index=False,
        )

    metadata = {
        "dataset_rows_N": int(len(df)),
        "input_features_P": int(len(input_cols)),
        "response_column": response_col,
        "desirability_type": "NTB",
        "desirability_target": config.target,
        "desirability_lower_observed": float(np.min(y)),
        "desirability_upper_observed": float(np.max(y)),
        "single_response_overall_D_equals_d": True,
        "prim_box_objective": "maximize mean NTB desirability D_bar",
        "peel_alpha": config.peel_alpha,
        "min_support": config.min_support,
        "subset_sizes": list(config.subset_sizes),
        "trials_per_subset": config.trials_per_subset,
        "total_subregions": len(boxes),
        "random_seed": config.random_seed,
        "hierarchical_linkage": "average",
        "empirical_distance": "1 - Jaccard(realized observation sets)",
        "n_memb_options": list(config.n_memb_options),
        "rho_limit": config.rho_limit,
        "k_grid": build_k_grid(len(boxes), config),
        "ecc_definition": "unweighted mean of effective-cluster compactness",
        "compactness_definition": "all-member intersection / all-member union",
        "important_adaptation": (
            "Concrete P=8 uses S={4,5,6,7}; these values are an adaptation "
            "chosen to retain four stochastic subspace settings and 2000 total "
            "trials, not values explicitly reported by the granularity paper."
        ),
    }

    with open(outdir / "00_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== NTB desirability ===")
    print(
        f"lower={np.min(y):.4f}, target={config.target:.4f}, "
        f"upper={np.max(y):.4f}"
    )
    print(f"Generated subregions: {len(boxes)}")
    print("\n=== Selected macrostructures ===")
    print(selected.to_string(index=False))
    print(f"\nOutputs saved to: {outdir.resolve()}")


# ---------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------

def _parse_subset_sizes(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("subset sizes cannot be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="concrete_dataset.csv",
        help="Path to concrete_dataset.csv",
    )
    parser.add_argument(
        "--out",
        default="baseline_ntb_stochastic_prim_output",
        help="Output directory",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=60.0,
        help="NTB target response value",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=100,
        help="Minimum valid PRIM box size",
    )
    parser.add_argument(
        "--subset-sizes",
        type=_parse_subset_sizes,
        default=(4, 5, 6, 7),
        help="Comma-separated random subspace sizes, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--trials-per-subset",
        type=int,
        default=500,
        help="Number of independent stochastic trials for each S",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Implementation random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        csv_path=args.csv,
        output_dir=args.out,
        target=args.target,
        min_support=args.min_support,
        subset_sizes=args.subset_sizes,
        trials_per_subset=args.trials_per_subset,
        random_seed=args.seed,
    )

    run_full_pipeline(cfg)
