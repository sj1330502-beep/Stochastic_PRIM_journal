"""
idea1_AB_integrated_originalpaper_score.py

아이디어 1번 통합 비교 코드
- 공통 전처리/클러스터링은 우리가 작성한 원 논문 baseline 로직을 기준으로 유지
- Concrete 단일 response -> NTB desirability(target=60) -> stochastic PRIM(mean D 최대화)
- 2000 subregions: S={4,5,6,7}, 각 500회
- empirical Jaccard distance -> average linkage
- effective-cluster heuristic: N_Memb={5,...,10}, rho_eff>=0.60,
  K={100,150,...,1000}, 각 N_Memb에서 ECC 최대 K 선택

아이디어 1 확장 부분
1) 하나의 recipe 후보를 만들기 위해 원 논문의 N_Memb별 최적 macrostructure 중
   ECC가 가장 높은 macrostructure 하나를 선택한다.
2) 그 macrostructure의 effective clusters 중
       ClusterScore = sqrt(mean_box_desirability * compactness)
   가 가장 높은 cluster를 선택한다.
   - member 수는 N_Memb 통과 여부에만 사용한다.
3) 동일한 선택 cluster / 동일한 통합박스에서 방법 A와 방법 B를 동시에 수행해
   동일한 confirmation set으로 공정하게 비교한다.

방법 A: Monte Carlo + multidimensional overlap density + RandomForest surrogate
방법 B: 실제 training 관측치만 이용한 2차 PRIM

주의:
- 원 논문 자체는 N_Memb 옵션들 가운데 단 하나를 고르는 규칙까지 제시하지 않는다.
  위의 'N_Memb별 최적 중 ECC 최대' 규칙은 아이디어 1에서 최종 recipe 하나를 만들기
  위해 기존 실험에서 사용해 온 확장 규칙이다.
- desirability의 lower/upper는 기존 실험과 동일하게 전체 concrete dataset의 관측 min/max를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import argparse

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor


HERE = Path(__file__).resolve().parent


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

@dataclass(frozen=True)
class Config:
    # I/O
    csv_path: str = str(HERE / "concrete_dataset.csv")
    output_dir: str = str(HERE / "idea1_AB_integrated_output")

    # Hold-out comparison (Idea 1 validation layer)
    train_ratio: float = 0.70
    n_random_seeds: int = 10
    master_split_seed: int = 20260824

    # Single-response NTB desirability
    target: float = 60.0

    # ----- Original-paper baseline settings retained -----
    peel_alpha: float = 0.05
    min_support: int = 100

    # Concrete adaptation already used in our baseline:
    # P=8 -> S<P, four settings x 500 = 2000 boxes.
    subset_sizes: tuple[int, ...] = (4, 5, 6, 7)
    trials_per_subset: int = 500

    n_memb_options: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
    rho_limit: float = 0.60
    k_grid_start: int = 100
    k_grid_stop: int = 1000
    k_grid_step: int = 50

    # Per-split stochastic PRIM seed; overwritten with split_seed in run_one_seed.
    random_seed: int = 2026

    # ----- Method A settings: keep final Method A logic -----
    mc_samples: int = 10000
    alpha_q: float = 5.0
    shrink_delta: float = 0.05
    boundary_band: float = 0.10
    eps: float = 0.05
    max_iter: int = 300
    n_min_ratio: float = 0.40
    overlap_probe_samples: int = 2000
    rf_estimators: int = 100

    # ----- Method B settings: keep final Method B logic -----
    ratio2_fixed: float = 0.40
    min_support2_floor: int = 20


@dataclass
class Box:
    trial_id: int
    subset_size: int
    selected_step: int
    bounds: np.ndarray
    observation_indices: np.ndarray
    mean_desirability: float
    mean_response: float
    min_response: float
    max_response: float


# =====================================================================
# 1. DATA + NTB DESIRABILITY
# =====================================================================

def load_concrete(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.shape[1] != 9:
        raise ValueError(
            "Expected 8 input columns + 1 response column for concrete data; "
            f"found {df.shape[1]} columns."
        )
    input_cols = list(df.columns[:8])
    response_col = df.columns[8]
    X = df[input_cols].to_numpy(dtype=float)
    y = df[response_col].to_numpy(dtype=float)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Missing values were detected in the dataset.")
    return df, input_cols, response_col, X, y


def make_desirability_function(y_all: np.ndarray, target: float = 60.0):
    """기존 실험과 동일한 triangular NTB desirability function."""
    y_all = np.asarray(y_all, dtype=float)
    lower, upper = float(y_all.min()), float(y_all.max())
    if not (lower < target < upper):
        raise ValueError(
            f"target must be inside observed range: lower={lower}, target={target}, upper={upper}"
        )

    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y, dtype=float)
        m1 = (y >= lower) & (y <= target)
        m2 = (y > target) & (y <= upper)
        d[m1] = (y[m1] - lower) / (target - lower)
        d[m2] = (upper - y[m2]) / (upper - target)
        return np.clip(d, 0.0, 1.0)

    return d_func, lower, upper


# =====================================================================
# 2. STOCHASTIC PRIM -- baseline logic
# =====================================================================

def mask_from_bounds(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.all((X >= bounds[:, 0]) & (X <= bounds[:, 1]), axis=1)


def _box_stats(y, desirability, mask):
    idx = np.flatnonzero(mask)
    yy = y[idx]
    dd = desirability[idx]
    return float(dd.mean()), float(yy.mean()), float(yy.min()), float(yy.max())


def _candidate_peel(
    X: np.ndarray,
    desirability: np.ndarray,
    current_mask: np.ndarray,
    current_bounds: np.ndarray,
    feature: int,
    side: str,
    peel_alpha: float,
):
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

    candidate_n = int(candidate_mask.sum())
    current_n = int(current_mask.sum())
    if candidate_n == 0 or candidate_n >= current_n:
        return None

    candidate_bounds = current_bounds.copy()
    if side == "lower":
        candidate_bounds[feature, 0] = max(candidate_bounds[feature, 0], q)
    else:
        candidate_bounds[feature, 1] = min(candidate_bounds[feature, 1], q)

    return {
        "objective": float(np.mean(desirability[candidate_mask])),
        "mask": candidate_mask,
        "bounds": candidate_bounds,
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
    n, p = X.shape
    if not (1 <= subset_size < p):
        raise ValueError(f"stochastic PRIM requires 1 <= S < P; got S={subset_size}, P={p}")

    current_mask = np.ones(n, dtype=bool)
    current_bounds = np.column_stack([X.min(axis=0), X.max(axis=0)]).astype(float)

    trajectory = [(0, current_mask.copy(), current_bounds.copy(), float(desirability.mean()))]
    step = 0

    while True:
        step += 1
        sampled_features = rng.choice(p, size=subset_size, replace=False)
        candidates = []
        for feature in sampled_features:
            lower = _candidate_peel(
                X, desirability, current_mask, current_bounds,
                int(feature), "lower", config.peel_alpha
            )
            upper = _candidate_peel(
                X, desirability, current_mask, current_bounds,
                int(feature), "upper", config.peel_alpha
            )
            if lower is not None:
                candidates.append(lower)
            if upper is not None:
                candidates.append(upper)

        if not candidates:
            break

        best_pos = int(np.argmax([c["objective"] for c in candidates]))
        best_candidate = candidates[best_pos]
        next_mask = best_candidate["mask"]

        # baseline paper-style stopping: first selected candidate below support is not retained.
        if int(next_mask.sum()) < config.min_support:
            break

        current_mask = next_mask
        current_bounds = best_candidate["bounds"]
        trajectory.append(
            (step, current_mask.copy(), current_bounds.copy(), float(best_candidate["objective"]))
        )

    best_idx = int(np.argmax([row[3] for row in trajectory]))
    selected_step, selected_mask, selected_bounds, _ = trajectory[best_idx]
    mean_d, mean_y, min_y, max_y = _box_stats(y, desirability, selected_mask)

    return Box(
        trial_id=trial_id,
        subset_size=subset_size,
        selected_step=int(selected_step),
        bounds=selected_bounds.copy(),
        observation_indices=np.flatnonzero(selected_mask),
        mean_desirability=mean_d,
        mean_response=mean_y,
        min_response=min_y,
        max_response=max_y,
    )


def generate_stochastic_prim_boxes(X, y, desirability, config: Config) -> list[Box]:
    rng = np.random.default_rng(config.random_seed)
    boxes: list[Box] = []
    trial_id = 0
    for S in config.subset_sizes:
        if S >= X.shape[1]:
            raise ValueError(f"S={S} must satisfy S<P={X.shape[1]}")
        for _ in range(config.trials_per_subset):
            trial_id += 1
            boxes.append(
                stochastic_prim_trial(X, y, desirability, S, trial_id, config, rng)
            )
    return boxes


# =====================================================================
# 3. EMPIRICAL JACCARD + AVERAGE-LINKAGE
# =====================================================================

def membership_matrix_from_boxes(boxes: Sequence[Box], X_reference: np.ndarray) -> np.ndarray:
    return np.vstack([mask_from_bounds(X_reference, b.bounds) for b in boxes]).astype(bool)


def empirical_jaccard_distance(membership: np.ndarray) -> np.ndarray:
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


def fit_average_linkage(distance_matrix: np.ndarray) -> np.ndarray:
    return linkage(squareform(distance_matrix, checks=False), method="average")


def exact_k_labels(Z: np.ndarray, k: int) -> np.ndarray:
    return cut_tree(Z, n_clusters=[k]).reshape(-1).astype(int)


# =====================================================================
# 4. EFFECTIVE CLUSTERS + ECC -- baseline logic
# =====================================================================

def cluster_compactness(member_rows: np.ndarray) -> float:
    if member_rows.ndim != 2 or member_rows.shape[0] == 0:
        raise ValueError("member_rows must contain at least one subregion")
    intersection_n = int(np.all(member_rows, axis=0).sum())
    union_n = int(np.any(member_rows, axis=0).sum())
    return float(intersection_n / union_n) if union_n > 0 else 0.0


def evaluate_partition(labels, membership, n_memb):
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

    comp = []
    eff_members = 0
    for lab in effective_labels:
        idx = np.flatnonzero(labels == lab)
        eff_members += int(idx.size)
        comp.append(cluster_compactness(membership[idx]))

    comp = np.asarray(comp, dtype=float)
    return {
        "K_actual": int(unique_labels.size),
        "N_eff": int(effective_labels.size),
        "N_eff_members": int(eff_members),
        "rho_eff": float(eff_members / membership.shape[0]),
        "ECC": float(comp.mean()),
        "ECC_min": float(comp.min()),
        "ECC_max": float(comp.max()),
    }


def build_k_grid(M: int, config: Config) -> list[int]:
    stop = min(config.k_grid_stop, M)
    grid = list(range(config.k_grid_start, stop + 1, config.k_grid_step))
    return grid if grid else [M]


def run_granularity_heuristic(Z, membership, config: Config):
    """각 N_Memb에 대해 원 논문식 ECC 최대화, rho_eff 제약 적용."""
    M = membership.shape[0]
    k_grid = build_k_grid(M, config)
    rows = []
    labels_cache = {}

    for K in k_grid:
        labels = exact_k_labels(Z, K)
        labels_cache[K] = labels
        for n_memb in config.n_memb_options:
            metrics = evaluate_partition(labels, membership, n_memb)
            rows.append({
                "N_Memb": n_memb,
                "K": K,
                **metrics,
                "feasible": metrics["rho_eff"] >= config.rho_limit,
            })

    trajectory = pd.DataFrame(rows)
    selected_rows = []
    selected_labels = {}

    for n_memb in config.n_memb_options:
        feasible = trajectory[
            (trajectory["N_Memb"] == n_memb)
            & trajectory["feasible"]
            & trajectory["ECC"].notna()
        ].copy()

        if feasible.empty:
            continue

        best_ecc = float(feasible["ECC"].max())
        tied = feasible[np.isclose(feasible["ECC"], best_ecc, atol=1e-12, rtol=0)]
        # baseline과 동일: exact tie이면 더 큰/rightmost K.
        best = tied.sort_values("K").iloc[-1]
        K_star = int(best["K"])
        selected_labels[n_memb] = labels_cache[K_star]
        selected_rows.append({
            "N_Memb": int(n_memb),
            "K_star": K_star,
            "N_eff": int(best["N_eff"]),
            "rho_eff": float(best["rho_eff"]),
            "ECC": float(best["ECC"]),
            "ECC_min": float(best["ECC_min"]),
            "ECC_max": float(best["ECC_max"]),
        })

    return trajectory, pd.DataFrame(selected_rows), selected_labels


def choose_one_macrostructure(selected_df: pd.DataFrame, selected_labels: dict[int, np.ndarray]):
    """
    Idea 1에서 최종 recipe 하나를 만들기 위한 확장 규칙.
    원 논문이 N_Memb별로 제시한 최적점들 중 ECC가 최대인 조합을 하나 선택한다.
    ECC tie이면 기존 실험과 일관되게 더 작은 N_Memb를 우선한다.
    """
    if selected_df.empty:
        return None
    tmp = selected_df.sort_values(["ECC", "N_Memb"], ascending=[False, True]).reset_index(drop=True)
    best = tmp.iloc[0]
    n_memb_star = int(best["N_Memb"])
    return best.to_dict(), selected_labels[n_memb_star]


# =====================================================================
# 5. NEW CLUSTER SELECTION: desirability + compactness score
# =====================================================================

def rank_effective_clusters_by_score(
    labels: np.ndarray,
    membership: np.ndarray,
    boxes: Sequence[Box],
    n_memb: int,
) -> pd.DataFrame:
    """
    member 수는 effective 여부(N_Memb) 판정에만 사용.
    최종 순위는 sqrt(mean_box_desirability * compactness)로 결정.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    effective_labels = unique_labels[counts >= n_memb]

    rows = []
    for lab in effective_labels:
        idx = np.flatnonzero(labels == lab)
        comp = cluster_compactness(membership[idx])
        mean_d = float(np.mean([boxes[i].mean_desirability for i in idx]))
        score = float(np.sqrt(max(mean_d, 0.0) * max(comp, 0.0)))
        rows.append({
            "cluster_id": int(lab),
            "member_subregions": int(idx.size),
            "mean_box_desirability": mean_d,
            "compactness": comp,
            "cluster_score": score,
            "member_indices": idx,
        })

    if not rows:
        return pd.DataFrame()

    # score tie -> meanD -> compactness -> member 수 순으로만 deterministic tie-break.
    return pd.DataFrame(rows).sort_values(
        ["cluster_score", "mean_box_desirability", "compactness", "member_subregions"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def unified_box_from_selected_cluster(boxes: Sequence[Box], member_indices: np.ndarray, X_train: np.ndarray):
    """
    기존 방법 A/B final 코드와 동일한 통합박스 구성:
    선택 cluster의 member subregions가 실제로 포함한 training 관측치를 모두 모아
    각 변수별 observed min/max로 하나의 통합 범위를 만든다.
    """
    all_idx = np.unique(np.concatenate([boxes[i].observation_indices for i in member_indices]))
    lo0 = X_train[all_idx].min(axis=0)
    hi0 = X_train[all_idx].max(axis=0)
    return lo0, hi0, all_idx


# =====================================================================
# 6. COMMON HOLD-OUT EVALUATION
# =====================================================================

def holdout_eval(lo, hi, X_confirm, D_confirm):
    mask = np.all((X_confirm >= lo) & (X_confirm <= hi), axis=1)
    n = int(mask.sum())
    d_mean = float(D_confirm[mask].mean()) if n > 0 else float("nan")
    return d_mean, n


# =====================================================================
# 7. METHOD A -- MC + overlap density + RF surrogate
# =====================================================================

def eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0, config: Config):
    mask = np.all((pts0 >= lo) & (pts0 <= hi), axis=1)
    n_pts = int(mask.sum())
    if n_pts < 20:
        return np.inf, n_pts
    dD = D_orig - np.percentile(dpred0[mask], config.alpha_q)
    return float(dD), n_pts


def virtual_overlap_density(
    lo, hi, member_los, member_his, p, face, rng, config: Config
):
    span = hi - lo
    if span[p] <= 1e-9:
        return np.inf

    band = config.boundary_band * span[p]
    lo_b, hi_b = lo.copy(), hi.copy()
    if face == "lo":
        hi_b[p] = lo[p] + band
    else:
        lo_b[p] = hi[p] - band

    pts = rng.uniform(lo_b, hi_b, size=(config.overlap_probe_samples, len(lo)))
    covered = np.zeros(config.overlap_probe_samples, dtype=bool)
    for m_lo, m_hi in zip(member_los, member_his):
        covered |= np.all((pts >= m_lo) & (pts <= m_hi), axis=1)
    return float(covered.mean())


def shrink_methodA(
    lo0, hi0, D_orig, member_los, member_his,
    surrogate, d_func, rng, X_train, config: Config
):
    lo, hi = lo0.copy(), hi0.copy()
    n0 = int(np.sum(np.all((X_train >= lo0) & (X_train <= hi0), axis=1)))
    n_min = max(1, int(n0 * config.n_min_ratio))

    # Final A improvement: MC points/predictions generated once, then filtered/reused.
    pts0 = rng.uniform(lo0, hi0, size=(config.mc_samples, len(lo0)))
    dpred0 = d_func(surrogate.predict(pts0))
    dD, _ = eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0, config)

    reason = "MAX_ITER"
    n_iter = 0

    for it in range(config.max_iter):
        n_iter = it
        n_remaining = int(np.sum(np.all((X_train >= lo) & (X_train <= hi), axis=1)))

        if dD <= config.eps:
            reason = "dD<=EPS"
            break
        if n_remaining <= n_min:
            reason = "n<=n_min"
            break

        best_face, best_density = None, np.inf
        for p in range(len(lo)):
            for face in ("lo", "hi"):
                dens = virtual_overlap_density(
                    lo, hi, member_los, member_his, p, face, rng, config
                )
                if dens < best_density:
                    best_density, best_face = dens, (p, face)

        if best_face is None or not np.isfinite(best_density):
            reason = "no_valid_face"
            break

        p, face = best_face
        step = config.shrink_delta * (hi[p] - lo[p])
        if face == "lo":
            lo[p] = min(lo[p] + step, hi[p] - 1e-9)
        else:
            hi[p] = max(hi[p] - step, lo[p] + 1e-9)

        dD, _ = eval_stage2_filtered(lo, hi, D_orig, pts0, dpred0, config)
    else:
        reason = "MAX_ITER"
        n_iter = config.max_iter

    n_remaining_final = int(np.sum(np.all((X_train >= lo) & (X_train <= hi), axis=1)))
    return lo, hi, reason, n_iter, float(dD), n_remaining_final, n_min, n0


# =====================================================================
# 8. METHOD B -- 2nd PRIM on observed training data only
# =====================================================================

def peel_within_box(X_sub, D_sub, lo0, hi0, min_support2, config: Config):
    P = X_sub.shape[1]
    idx = np.arange(len(D_sub))
    lo, hi = lo0.copy(), hi0.copy()
    best_lo, best_hi, best_obj = lo.copy(), hi.copy(), float(D_sub.mean())

    while True:
        if len(idx) * (1 - config.peel_alpha) < min_support2:
            break

        cand_keep = None
        cand_obj = -np.inf
        cand_p = None
        cand_bound = None

        # deterministic 2nd PRIM: all 8 features x 2 directions.
        for p in range(P):
            xp = X_sub[idx, p]
            lo_q = np.quantile(xp, config.peel_alpha)
            hi_q = np.quantile(xp, 1 - config.peel_alpha)

            for keep, bound in (
                (idx[xp > lo_q], ("lo", lo_q)),
                (idx[xp < hi_q], ("hi", hi_q)),
            ):
                if min_support2 <= len(keep) < len(idx):
                    obj = float(D_sub[keep].mean())
                    if obj > cand_obj:
                        cand_obj, cand_keep, cand_p, cand_bound = obj, keep, p, bound

        if cand_keep is None:
            break

        idx = cand_keep
        side, val = cand_bound
        if side == "lo":
            lo[cand_p] = val
        else:
            hi[cand_p] = val

        if cand_obj > best_obj:
            best_obj = cand_obj
            best_lo, best_hi = lo.copy(), hi.copy()

    return best_lo, best_hi, best_obj


# =====================================================================
# 9. ONE SPLIT: SAME BASELINE -> SAME CLUSTER -> A vs B
# =====================================================================

def run_one_seed(X_all, y_all, input_cols, split_seed: int, config: Config):
    rng_split = np.random.default_rng(split_seed)
    n = len(y_all)
    perm = rng_split.permutation(n)
    n_train = int(n * config.train_ratio)
    train_idx, confirm_idx = perm[:n_train], perm[n_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_confirm, y_confirm = X_all[confirm_idx], y_all[confirm_idx]

    # Keep prior experiment's desirability definition based on full-data observed lower/upper.
    d_func, lower, upper = make_desirability_function(y_all, config.target)
    D_train = d_func(y_train)
    D_confirm = d_func(y_confirm)

    # Same original-paper baseline for both A and B.
    split_cfg = replace(config, random_seed=int(split_seed))
    boxes = generate_stochastic_prim_boxes(X_train, y_train, D_train, split_cfg)
    membership = membership_matrix_from_boxes(boxes, X_train)
    dist = empirical_jaccard_distance(membership)
    Z = fit_average_linkage(dist)

    trajectory, selected_df, selected_labels = run_granularity_heuristic(Z, membership, split_cfg)
    macro = choose_one_macrostructure(selected_df, selected_labels)
    if macro is None:
        return None, trajectory, selected_df, None, []

    macro_row, labels = macro
    n_memb_star = int(macro_row["N_Memb"])
    K_star = int(macro_row["K_star"])

    ranked = rank_effective_clusters_by_score(labels, membership, boxes, n_memb_star)
    if ranked.empty:
        return None, trajectory, selected_df, ranked, []

    best_cluster = ranked.iloc[0]
    mem = best_cluster["member_indices"]

    lo0, hi0, union_obs_idx = unified_box_from_selected_cluster(boxes, mem, X_train)
    d_before, n_before = holdout_eval(lo0, hi0, X_confirm, D_confirm)

    # Cluster's mean subregion desirability: same quantity used in previous Method A.
    D_orig = float(best_cluster["mean_box_desirability"])

    # ---------------- METHOD A ----------------
    surrogate = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        random_state=int(split_seed),
    ).fit(X_train, y_train)

    rng_a = np.random.default_rng(int(split_seed) + 500000)
    member_los = np.array([X_train[boxes[i].observation_indices].min(axis=0) for i in mem])
    member_his = np.array([X_train[boxes[i].observation_indices].max(axis=0) for i in mem])

    lo_a, hi_a, reason_a, n_iter_a, dD_a, n_remaining_a, n_min_a, n0_a = shrink_methodA(
        lo0, hi0, D_orig, member_los, member_his,
        surrogate, d_func, rng_a, X_train, split_cfg
    )
    d_after_a, n_after_a = holdout_eval(lo_a, hi_a, X_confirm, D_confirm)

    # ---------------- METHOD B ----------------
    in_box = np.all((X_train >= lo0) & (X_train <= hi0), axis=1)
    X_sub, D_sub = X_train[in_box], D_train[in_box]
    n_box = int(in_box.sum())
    min_support2 = max(split_cfg.min_support2_floor, int(n_box * split_cfg.ratio2_fixed))
    lo_b, hi_b, train_d_after_b = peel_within_box(
        X_sub, D_sub, lo0, hi0, min_support2, split_cfg
    )
    d_after_b, n_after_b = holdout_eval(lo_b, hi_b, X_confirm, D_confirm)

    result = {
        "split_seed": int(split_seed),
        "n_train": int(len(train_idx)),
        "n_confirm": int(len(confirm_idx)),
        "generated_boxes": int(len(boxes)),
        "N_Memb_star": n_memb_star,
        "K_star": K_star,
        "N_eff": int(macro_row["N_eff"]),
        "rho_eff": float(macro_row["rho_eff"]),
        "ECC": float(macro_row["ECC"]),
        "selected_cluster_id": int(best_cluster["cluster_id"]),
        "cluster_members": int(best_cluster["member_subregions"]),
        "cluster_meanD": float(best_cluster["mean_box_desirability"]),
        "cluster_compactness": float(best_cluster["compactness"]),
        "cluster_score": float(best_cluster["cluster_score"]),
        "unified_train_observations": int(len(union_obs_idx)),
        "d_before": d_before,
        "n_before": int(n_before),
        "d_after_A": d_after_a,
        "n_after_A": int(n_after_a),
        "improve_A": d_after_a - d_before if np.isfinite(d_after_a) and np.isfinite(d_before) else np.nan,
        "A_reason": reason_a,
        "A_iterations": int(n_iter_a),
        "A_dD": float(dD_a),
        "A_train_remaining": int(n_remaining_a),
        "A_train_min": int(n_min_a),
        "A_train_start": int(n0_a),
        "d_after_B": d_after_b,
        "n_after_B": int(n_after_b),
        "improve_B": d_after_b - d_before if np.isfinite(d_after_b) and np.isfinite(d_before) else np.nan,
        "B_train_box_n": int(n_box),
        "B_min_support2": int(min_support2),
        "B_train_bestD": float(train_d_after_b),
        "A_minus_B_D": d_after_a - d_after_b if np.isfinite(d_after_a) and np.isfinite(d_after_b) else np.nan,
        "A_minus_B_n": int(n_after_a - n_after_b),
    }

    bounds = []
    for j, name in enumerate(input_cols):
        bounds.append({
            "split_seed": int(split_seed),
            "variable": name,
            "before_lower": float(lo0[j]),
            "before_upper": float(hi0[j]),
            "A_lower": float(lo_a[j]),
            "A_upper": float(hi_a[j]),
            "B_lower": float(lo_b[j]),
            "B_upper": float(hi_b[j]),
        })

    return result, trajectory, selected_df, ranked, bounds


# =====================================================================
# 10. RUN + SAVE COMPARISON
# =====================================================================

def _mean_std(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.mean(x)), float(np.std(x, ddof=0))


def run(config: Config):
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, input_cols, response_col, X_all, y_all = load_concrete(config.csv_path)
    _, lower, upper = make_desirability_function(y_all, config.target)

    master_rng = np.random.default_rng(config.master_split_seed)
    split_seeds = [int(x) for x in master_rng.integers(0, 1_000_000, size=config.n_random_seeds)]

    print("=" * 100)
    print(" IDEA 1 -- original-paper baseline + score-based cluster selection + Method A/B integrated comparison")
    print("=" * 100)
    print(f"CSV: {config.csv_path}")
    print(f"Response: {response_col}")
    print(f"NTB: lower={lower:.4f}, target={config.target:.4f}, upper={upper:.4f}")
    print(f"PRIM: alpha={config.peel_alpha}, min_support={config.min_support}, "
          f"S={config.subset_sizes}, T={config.trials_per_subset} -> "
          f"{len(config.subset_sizes) * config.trials_per_subset} boxes/split")
    print(f"Granularity: N_Memb={config.n_memb_options}, rho_limit={config.rho_limit}, "
          f"K={config.k_grid_start}:{config.k_grid_step}:{config.k_grid_stop}")
    print(f"Cluster selection: sqrt(meanD * compactness)")
    print(f"Split seeds: {split_seeds}\n")

    results = []
    all_bounds = []
    all_macro = []
    all_ranked = []

    for i, seed in enumerate(split_seeds, start=1):
        print(f"[{i}/{len(split_seeds)}] split_seed={seed} -- running common baseline ...")
        out = run_one_seed(X_all, y_all, input_cols, seed, config)
        if len(out) == 4:
            result, trajectory, selected_df, ranked = out
            bounds = []
        else:
            result, trajectory, selected_df, ranked, bounds = out

        if selected_df is not None and not selected_df.empty:
            tmp = selected_df.copy()
            tmp.insert(0, "split_seed", seed)
            all_macro.append(tmp)

        if ranked is not None and not ranked.empty:
            # member_indices is ndarray and is not needed in CSV output.
            tmp = ranked.drop(columns=["member_indices"]).copy()
            tmp.insert(0, "split_seed", seed)
            all_ranked.append(tmp)

        if result is None:
            print("  -> no feasible macrostructure/cluster; skipped\n")
            continue

        results.append(result)
        all_bounds.extend(bounds)

        print(
            f"  macro: N_Memb*={result['N_Memb_star']}, K*={result['K_star']}, "
            f"ECC={result['ECC']:.4f}, rho={result['rho_eff']:.4f}"
        )
        print(
            f"  cluster(score): id={result['selected_cluster_id']}, "
            f"members={result['cluster_members']}, meanD={result['cluster_meanD']:.4f}, "
            f"compactness={result['cluster_compactness']:.4f}, score={result['cluster_score']:.4f}"
        )
        print(f"  BEFORE : D_conf={result['d_before']:.4f} (n={result['n_before']})")
        print(
            f"  METHOD A: D_conf={result['d_after_A']:.4f} (n={result['n_after_A']}), "
            f"improve={result['improve_A']:+.4f}, stop={result['A_reason']}"
        )
        print(
            f"  METHOD B: D_conf={result['d_after_B']:.4f} (n={result['n_after_B']}), "
            f"improve={result['improve_B']:+.4f}"
        )
        print()

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No valid comparison results were produced.")
        return

    results_path = outdir / "AB_comparison_by_seed.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    if all_bounds:
        pd.DataFrame(all_bounds).to_csv(
            outdir / "AB_selected_recipe_bounds_by_seed.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if all_macro:
        pd.concat(all_macro, ignore_index=True).to_csv(
            outdir / "macrostructure_candidates_by_seed.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if all_ranked:
        pd.concat(all_ranked, ignore_index=True).to_csv(
            outdir / "effective_cluster_score_ranking_by_seed.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 100)
    print(" SUMMARY")
    print("=" * 100)

    metrics = [
        ("Before D_conf", "d_before"),
        ("Method A D_conf", "d_after_A"),
        ("Method B D_conf", "d_after_B"),
        ("Method A improvement", "improve_A"),
        ("Method B improvement", "improve_B"),
        ("Before n_conf", "n_before"),
        ("Method A n_conf", "n_after_A"),
        ("Method B n_conf", "n_after_B"),
    ]
    summary_rows = []
    for label, col in metrics:
        mean, std = _mean_std(results_df[col])
        summary_rows.append({"metric": label, "mean": mean, "std": std})
        print(f"{label:<24}: mean={mean:.4f}, std={std:.4f}")

    valid_pair = results_df[["d_after_A", "d_after_B"]].dropna()
    if not valid_pair.empty:
        A_wins = int((valid_pair["d_after_A"] > valid_pair["d_after_B"]).sum())
        B_wins = int((valid_pair["d_after_B"] > valid_pair["d_after_A"]).sum())
        ties = int(np.isclose(valid_pair["d_after_A"], valid_pair["d_after_B"]).sum())
        print(f"\nD_conf wins: A={A_wins}, B={B_wins}, ties={ties}")

    pd.DataFrame(summary_rows).to_csv(
        outdir / "AB_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {results_path}")
    print(f"Output directory: {outdir}")


# =====================================================================
# 11. CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(HERE / "concrete_dataset.csv"))
    p.add_argument("--output-dir", default=str(HERE / "idea1_AB_integrated_output"))
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--master-seed", type=int, default=20260824)
    p.add_argument("--trials-per-subset", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_random_seeds=args.n_seeds,
        master_split_seed=args.master_seed,
        trials_per_subset=args.trials_per_subset,
    )
    run(cfg)
