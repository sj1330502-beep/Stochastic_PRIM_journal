"""
================================================================================
아이디어 6번 — Naval 데이터에서 원 논문 방식과 방식 4 비교
================================================================================
두 방식이 완전히 같은 training split, MRS-PRIM 박스, Jaccard linkage,
N_Memb x K 탐색 격자를 사용하도록 한 공정 비교 스크립트다.

원 논문 방식
  · rho_eff >= 0.60인 후보 중 ECC가 가장 큰 (N_Memb*, K*) 선택

방식 4
  · score = ECC^a * rho_eff^b * Dbar_largest^c
  · Dbar_largest는 가장 큰 유효 클러스터에 속한 member box들의
    training 평균 desirability다.
  · 기본 가중치는 canonical 구현과 같은 a=0.2, b=0.2, c=99.0이다.

데이터 누출 방지
  · 반응별 desirability 경계는 training 데이터에서만 추정한다.
  · 두 방식의 선택이 모두 끝난 후 untouched confirmation을 한 번 평가한다.
  · confirmation 결과로 방식 4 가중치를 탐색하거나 다시 선택하지 않는다.

실행 예시
  python idea6/idea6_naval_original_vs_method4.py
  python idea6/idea6_naval_original_vs_method4.py --a 0.2 --b 0.2 --c 99
================================================================================
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


CSV_PATH = Path(__file__).resolve().parents[1] / "naval_propulsion_dataset.csv"

RESPONSE_COLS = [
    "GT_Compressor_decay_state_coefficient",
    "GT_Turbine_decay_state_coefficient",
]
DROP_COLS = [
    "GT_Compressor_inlet_air_temp_T1",      # 상수
    "GT_Compressor_inlet_air_pressure_P1",  # 상수
    "Port_Propeller_Torque_Tp",             # Starboard torque와 중복
]

TRAIN_RATIO = 0.70
SPLIT_SEED = 0

ALPHA_PEEL = 0.05
MIN_SUPPORT = 250
S_OPTIONS = (7, 9, 10)
T_PER_SIZE = 250
SEED_PRIM = 1

N_MEMB_OPTIONS = tuple(range(5, 11))
K_GRID_STEP = 20
RHO_LIMIT = 0.60

METHOD4_DEFAULTS = dict(a=0.0, b=0.0, c=1.0)


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("가중치 지수는 0 이상이어야 합니다.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Naval 데이터에서 원 논문 방식과 방식 4를 hold-out 비교합니다."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Naval CSV 경로 (기본값: 저장소 루트의 naval_propulsion_dataset.csv)",
    )
    parser.add_argument("--a", type=nonnegative_float, default=METHOD4_DEFAULTS["a"],
                        help="방식 4 ECC 지수 (기본값: 0.2)")
    parser.add_argument("--b", type=nonnegative_float, default=METHOD4_DEFAULTS["b"],
                        help="방식 4 rho_eff 지수 (기본값: 0.2)")
    parser.add_argument("--c", type=nonnegative_float, default=METHOD4_DEFAULTS["c"],
                        help="방식 4 Dbar_largest 지수 (기본값: 99.0)")
    args = parser.parse_args()
    if args.a == args.b == args.c == 0:
        parser.error("a, b, c를 모두 0으로 둘 수는 없습니다.")
    return args


# ============================================= 데이터와 desirability
def split_indices(n, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_ratio)
    if n_train == 0 or n_train == n:
        raise ValueError("training과 confirmation에 각각 관측치가 필요합니다.")
    return perm[:n_train], perm[n_train:]


def fit_desirability_bounds(df, train_idx):
    """반응별 LTB 경계를 training에서만 추정한다."""
    bounds = []
    for col in RESPONSE_COLS:
        y_train = df[col].to_numpy(dtype=float)[train_idx]
        bounds.append((float(y_train.min()), float(y_train.max())))
    return bounds


def desirability_from_bounds(df, indices, bounds):
    """두 LTB desirability를 원 논문 방식의 기하평균으로 결합한다."""
    components = []
    for col, (lower, upper) in zip(RESPONSE_COLS, bounds):
        y = df[col].to_numpy(dtype=float)[indices]
        if upper > lower:
            d = (y - lower) / (upper - lower)
        else:
            d = np.ones_like(y)
        components.append(np.clip(d, 0.0, 1.0))

    d_matrix = np.column_stack(components)
    return np.prod(d_matrix, axis=1) ** (1.0 / d_matrix.shape[1])


def load_naval(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Naval CSV를 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = set(RESPONSE_COLS + DROP_COLS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    feature_names = [
        col for col in df.columns
        if col not in RESPONSE_COLS and col not in DROP_COLS
    ]
    X = df[feature_names].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("입력 변수에 결측치 또는 무한대가 있습니다.")
    for col in RESPONSE_COLS:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"반응 변수 {col}에 결측치 또는 무한대가 있습니다.")
    return df, X, feature_names


# ============================================= MRS-PRIM 박스 생성 (training만 사용)
def peel_trajectory(X, D, subset_size, rng):
    n_features = X.shape[1]
    idx = np.arange(len(D))
    best_idx = idx.copy()
    best_obj = float(D.mean())

    while True:
        if len(idx) * (1.0 - ALPHA_PEEL) < MIN_SUPPORT:
            break

        features = rng.choice(n_features, size=subset_size, replace=False)
        candidate_idx = None
        candidate_obj = -np.inf

        for feature in features:
            values = X[idx, feature]
            lower_q = np.quantile(values, ALPHA_PEEL)
            upper_q = np.quantile(values, 1.0 - ALPHA_PEEL)
            for keep in (idx[values > lower_q], idx[values < upper_q]):
                if MIN_SUPPORT <= len(keep) < len(idx):
                    obj = float(D[keep].mean())
                    if obj > candidate_obj:
                        candidate_obj = obj
                        candidate_idx = keep

        if candidate_idx is None:
            break

        idx = candidate_idx
        if candidate_obj > best_obj:
            best_obj = candidate_obj
            best_idx = idx.copy()

    return best_idx, best_obj


def build_boxes(X, D):
    rng = np.random.default_rng(SEED_PRIM)
    boxes = []
    total = len(S_OPTIONS) * T_PER_SIZE
    completed = 0

    for subset_size in S_OPTIONS:
        if subset_size > X.shape[1]:
            raise ValueError(
                f"S={subset_size}는 입력 변수 수 {X.shape[1]}보다 클 수 없습니다."
            )
        for _ in range(T_PER_SIZE):
            idx, dbar = peel_trajectory(X, D, subset_size, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({
                    "idx": idx,
                    "support": len(idx),
                    "dbar": dbar,
                })
            completed += 1
            if completed % 100 == 0:
                print(f"    trial {completed}/{total} | 박스 {len(boxes)}개", flush=True)

    if len(boxes) < 2:
        raise RuntimeError("계층적 군집화에 필요한 MRS-PRIM 박스가 2개 미만입니다.")
    return boxes


# ============================================= 공통 Jaccard linkage와 탐색 후보
def build_linkage(boxes, n_observations):
    """모든 방식이 공유할 empirical Jaccard average-linkage를 한 번 만든다."""
    n_boxes = len(boxes)
    membership = np.zeros((n_boxes, n_observations), dtype=np.int32)
    for i, box in enumerate(boxes):
        membership[i, box["idx"]] = 1

    intersection = (membership @ membership.T).astype(float)
    support = np.array([box["support"] for box in boxes], dtype=float)
    union = support[:, None] + support[None, :] - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    observation_sets = [set(box["idx"].tolist()) for box in boxes]
    hierarchy = linkage(squareform(distance, checks=False), method="average")
    return hierarchy, observation_sets


def make_k_grid(n_boxes):
    lower = max(int(0.05 * n_boxes), 5)
    upper = min(int(0.50 * n_boxes), n_boxes - 1)
    if lower > upper:
        raise RuntimeError("현재 박스 수로는 K 탐색 격자를 만들 수 없습니다.")

    grid = list(range(lower, upper + 1, K_GRID_STEP))
    if not grid:
        grid = [lower]
    return grid


def precompute_raw_clusters(hierarchy, observation_sets, boxes, k_grid):
    """K별 원본 클러스터를 한 번 계산해 두 선택 방식이 그대로 공유한다."""
    raw_by_k = {}
    for K in k_grid:
        labels = fcluster(hierarchy, K, criterion="maxclust")
        clusters = []
        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            member_sets = [observation_sets[i] for i in members]
            intersection = set.intersection(*member_sets)
            union = set.union(*member_sets)
            gamma = len(intersection) / len(union) if union else 0.0
            dbar = float(np.mean([boxes[i]["dbar"] for i in members]))
            clusters.append({
                "members": members,
                "n_members": len(members),
                "gamma": gamma,
                "dbar": dbar,
            })
        raw_by_k[K] = clusters
    return raw_by_k


def evaluate_candidate(raw_clusters, n_memb, n_boxes):
    effective = [c for c in raw_clusters if c["n_members"] >= n_memb]
    if not effective:
        return None

    rho_eff = sum(c["n_members"] for c in effective) / n_boxes
    ecc = float(np.mean([c["gamma"] for c in effective]))
    largest = max(effective, key=lambda c: c["n_members"])
    return {
        "ecc": ecc,
        "rho_eff": rho_eff,
        "n_eff": len(effective),
        "dbar_largest": largest["dbar"],
        "largest_members": largest["members"],
        "largest_n_members": largest["n_members"],
    }


def build_shared_candidates(raw_by_k, n_boxes, k_grid):
    candidates = []
    for n_memb in N_MEMB_OPTIONS:
        for K in k_grid:
            metrics = evaluate_candidate(raw_by_k[K], n_memb, n_boxes)
            if metrics is not None:
                candidates.append({"n_memb": n_memb, "K": K, **metrics})
    if not candidates:
        raise RuntimeError("평가 가능한 (N_Memb, K) 후보가 없습니다.")
    return candidates


# ============================================= 원 논문 방식과 방식 4 선택
def select_original(candidates, rho_limit=RHO_LIMIT):
    feasible = [c for c in candidates if c["rho_eff"] >= rho_limit]
    if not feasible:
        raise RuntimeError(
            f"원 논문 조건 rho_eff >= {rho_limit:.2f}를 만족하는 후보가 없습니다."
        )
    # 동률이면 더 작은 N_Memb, 더 작은 K를 택해 결과를 결정적으로 만든다.
    return max(feasible, key=lambda c: (c["ecc"], -c["n_memb"], -c["K"]))


def method4_score(candidate, a, b, c):
    dbar = max(candidate["dbar_largest"], 1e-12)
    return (
        candidate["ecc"] ** a
        * candidate["rho_eff"] ** b
        * dbar ** c
    )


def select_method4(candidates, a, b, c):
    scored = []
    for candidate in candidates:
        scored.append({
            **candidate,
            "method4_score": method4_score(candidate, a, b, c),
        })
    return max(
        scored,
        key=lambda item: (
            item["method4_score"],
            -item["n_memb"],
            -item["K"],
        ),
    )


# ============================================= untouched confirmation 평가
def confirmation_evaluation(boxes, selected, X_train, X_confirm, D_confirm):
    member_box_ids = selected["largest_members"]
    union_idx = np.unique(np.concatenate([boxes[i]["idx"] for i in member_box_ids]))
    lower = X_train[union_idx].min(axis=0)
    upper = X_train[union_idx].max(axis=0)

    mask = np.all((X_confirm >= lower) & (X_confirm <= upper), axis=1)
    n_confirm = int(mask.sum())
    d_confirm = float(D_confirm[mask].mean()) if n_confirm else float("nan")
    coverage = n_confirm / len(X_confirm)
    return {
        "d_confirm": d_confirm,
        "n_confirm": n_confirm,
        "confirm_coverage": coverage,
        "lower": lower,
        "upper": upper,
    }


def format_float(value, decimals=4):
    return "NA" if not np.isfinite(value) else f"{value:.{decimals}f}"


def print_comparison(rows):
    print("\n" + "=" * 124)
    print("  원 논문 방식 vs 방식 4 — untouched confirmation 비교")
    print("=" * 124)
    header = (
        f'{"방식":>18} {"N_Memb*":>8} {"K*":>6} {"N_eff":>7} '
        f'{"ECC":>9} {"rho_eff":>9} {"Dbar_train":>12} {"score4":>12} '
        f'{"box_mem":>8} {"n_conf":>8} {"coverage":>10} {"D_conf":>10}'
    )
    print(header)

    for row in rows:
        selected = row["selected"]
        confirmation = row["confirmation"]
        score = selected.get("method4_score", float("nan"))
        print(
            f'{row["name"]:>18} '
            f'{selected["n_memb"]:>8} '
            f'{selected["K"]:>6} '
            f'{selected["n_eff"]:>7} '
            f'{selected["ecc"]:>9.4f} '
            f'{selected["rho_eff"]:>9.3f} '
            f'{selected["dbar_largest"]:>12.4f} '
            f'{format_float(score, 6):>12} '
            f'{selected["largest_n_members"]:>8} '
            f'{confirmation["n_confirm"]:>8} '
            f'{confirmation["confirm_coverage"]:>10.3%} '
            f'{format_float(confirmation["d_confirm"]):>10}'
        )


def print_envelope(feature_names, rows):
    print("\n선택된 최대 유효 클러스터의 축정렬 운전 범위(training에서 구성)")
    for row in rows:
        print(f'\n[{row["name"]}]')
        lower = row["confirmation"]["lower"]
        upper = row["confirmation"]["upper"]
        for name, lo, hi in zip(feature_names, lower, upper):
            print(f"  {name:>38}: [{lo:>12.5f}, {hi:>12.5f}]")


# ============================================= 실행
def main():
    args = parse_args()
    df, X_all, feature_names = load_naval(args.csv)
    train_idx, confirm_idx = split_indices(len(df))

    bounds = fit_desirability_bounds(df, train_idx)
    D_train = desirability_from_bounds(df, train_idx, bounds)
    D_confirm = desirability_from_bounds(df, confirm_idx, bounds)
    X_train = X_all[train_idx]
    X_confirm = X_all[confirm_idx]

    print(
        f"[0] Naval {len(df)}행, 입력 {len(feature_names)}개, 반응 2개 | "
        f"training {len(train_idx)}개 / confirmation {len(confirm_idx)}개"
    )
    print("    desirability 경계는 training에서만 추정")
    for col, (lower, upper) in zip(RESPONSE_COLS, bounds):
        print(f"    {col}: [{lower:.6f}, {upper:.6f}]")

    print("\n[1] 공통 training 데이터로 MRS-PRIM 박스 생성")
    boxes = build_boxes(X_train, D_train)
    n_boxes = len(boxes)
    print(f"    생성된 공통 박스 M={n_boxes}")

    print("\n[2] 공통 Jaccard average-linkage와 (N_Memb x K) 후보 계산")
    hierarchy, observation_sets = build_linkage(boxes, len(X_train))
    k_grid = make_k_grid(n_boxes)
    raw_by_k = precompute_raw_clusters(
        hierarchy, observation_sets, boxes, k_grid
    )
    candidates = build_shared_candidates(raw_by_k, n_boxes, k_grid)
    print(
        f"    K={k_grid[0]}~{k_grid[-1]} (간격 {K_GRID_STEP}), "
        f"N_Memb={N_MEMB_OPTIONS[0]}~{N_MEMB_OPTIONS[-1]}, "
        f"공통 후보 {len(candidates)}개"
    )

    print("\n[3] confirmation을 보지 않고 두 방식 선택")
    original = select_original(candidates)
    method4 = select_method4(candidates, args.a, args.b, args.c)
    print(f"    원 논문: rho_eff >= {RHO_LIMIT:.2f} 중 ECC 최대")
    print(
        f"    방식 4: ECC^{args.a:g} x rho_eff^{args.b:g} "
        f"x Dbar_largest^{args.c:g} 최대"
    )

    print("\n[4] 선택 완료 후 untouched confirmation을 한 번 평가")
    rows = []
    for name, selected in (("원 논문", original), ("방식 4", method4)):
        confirmation = confirmation_evaluation(
            boxes, selected, X_train, X_confirm, D_confirm
        )
        rows.append({
            "name": name,
            "selected": selected,
            "confirmation": confirmation,
        })

    print_comparison(rows)
    print_envelope(feature_names, rows)

    original_d = rows[0]["confirmation"]["d_confirm"]
    method4_d = rows[1]["confirmation"]["d_confirm"]
    print("\n" + "=" * 124)
    if np.isfinite(original_d) and np.isfinite(method4_d):
        print(
            f"confirmation D 차이(방식 4 - 원 논문) = "
            f"{method4_d - original_d:+.4f}"
        )
    else:
        print("한 방식 이상의 confirmation 범위가 비어 D 차이를 계산하지 못했습니다.")
    print(
        "주의: 이 hold-out은 사전에 고정한 두 방식을 비교하는 최종 평가이며, "
        "결과를 보고 a/b/c를 다시 고르면 별도 validation이 필요합니다."
    )
    print("완료.")


if __name__ == "__main__":
    main()
