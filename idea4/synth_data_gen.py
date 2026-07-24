"""
================================================================================
합성 데이터 생성 — 정답 기반 검증용 (Synthetic Ground-Truth Data)
================================================================================
목적
  다중 high-performing region의 개수와 위치를 미리 알고 있는 합성 데이터를
  생성한다. 콘크리트 데이터셋의 실제 변수 범위(스케일)를 그대로 사용하되,
  desirability 는 우리가 심어놓은 G개의 정답 중심(가우시안 봉우리) 기반으로
  새로 계산한다. 관측치 값 자체도 새로 무작위 생성하며, 콘크리트의 실제
  압축강도/desirability는 전혀 사용하지 않는다.

구조
  D(x) = baseline + sum_g[ A_g * exp(-||x-c_g||^2 / (2*sigma_g^2)) ] + noise

  x           : 입력 벡터 (P차원, 각 변수는 콘크리트 실제 범위에서 균등 샘플)
  c_g         : g번째 정답 영역의 중심 (우리가 미리 정함 = 정답 위치)
  A_g         : g번째 정답 영역의 세기 (상대적 desirability 수준)
  sigma_g     : g번째 정답 영역의 폭 (넓을수록 완만하고 넓은 언덕)
  거리 계산은 변수별로 실제 스케일이 다르므로, 정규화된 거리(각 변수를
  자기 범위로 나눈 거리)를 사용해 특정 변수가 거리 계산을 지배하지 않게 한다.

난이도 조절 변수 (모두 함수 인자로 노출)
  G (정답 개수), 중심 간 거리, A_g(세기), noise 수준, P(차원 - 콘크리트는 8 고정)
================================================================================
"""
import numpy as np
import pandas as pd

# 콘크리트 데이터의 실제 변수 범위 (참고용 기본값 — 실제 실행 시 CSV에서 재확인 권장)
CONCRETE_RANGES = {
    'Cement':     (102.0, 540.0),
    'Slag':       (0.0,   359.4),
    'FlyAsh':     (0.0,   200.1),
    'Water':      (121.8, 247.0),
    'Superplast': (0.0,   32.2),
    'CoarseAgg':  (801.0, 1145.0),
    'FineAgg':    (594.0, 992.6),
    'Age':        (1.0,   365.0),
}
FEAT_NAMES = list(CONCRETE_RANGES.keys())
P_DEFAULT = len(FEAT_NAMES)


def load_real_ranges(csv_path):
    """실제 콘크리트 CSV에서 변수별 (min, max) 를 다시 확인해 사용 (권장)."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    ranges = {}
    for i, name in enumerate(FEAT_NAMES):
        ranges[name] = (float(X[:, i].min()), float(X[:, i].max()))
    return ranges


def sample_centers(ranges, G, min_sep_ratio=0.15, seed=0):
    """
    정답 중심 G개를 각 변수 범위 안에서 무작위로 뽑되, 서로 너무 가깝지
    않도록(min_sep_ratio: 정규화 거리 기준 최소 간격) 보정한다.
    """
    rng = np.random.default_rng(seed)
    P = len(ranges)
    lo = np.array([ranges[n][0] for n in FEAT_NAMES[:P]])
    hi = np.array([ranges[n][1] for n in FEAT_NAMES[:P]])

    centers = []
    attempts = 0
    while len(centers) < G and attempts < 5000:
        cand = rng.uniform(lo, hi)
        ok = True
        for c in centers:
            # 정규화 거리(각 변수를 자기 범위 폭으로 나눔)
            d = np.sqrt(np.mean(((cand - c) / (hi - lo)) ** 2))
            if d < min_sep_ratio:
                ok = False
                break
        if ok:
            centers.append(cand)
        attempts += 1
    if len(centers) < G:
        raise RuntimeError(f'min_sep_ratio={min_sep_ratio} 로 {G}개 중심을 '
                           f'배치하지 못했습니다. 값을 낮추거나 G를 줄이세요.')
    return np.array(centers)


def generate_synthetic_data(
    ranges,               # dict: 변수명 -> (min, max), 콘크리트 실제 범위
    N=1000,                # 관측치 수
    G=3,                   # 정답 영역 개수
    A=None,                 # 각 영역 세기 (None이면 전부 0.7로 동일)
    sigma_ratio=0.15,       # 언덕 폭 (변수 범위 대비 비율, 정규화 거리 기준)
    noise_std=0.05,         # 가우시안 노이즈 표준편차
    baseline=0.05,          # 기본 desirability
    min_sep_ratio=0.15,     # 중심 간 최소 정규화 거리
    seed=0,
):
    """
    반환: X (N,P) 관측치, D (N,) desirability, centers (G,P) 정답 위치,
          membership (N,) 각 관측치가 가장 가까운 정답 영역 인덱스(참고용,
          '소속'의 엄밀한 정의는 검증 단계에서 반경 기준으로 별도 처리 권장)
    """
    rng = np.random.default_rng(seed)
    P = len(ranges)
    feat_names = FEAT_NAMES[:P]
    lo = np.array([ranges[n][0] for n in feat_names])
    hi = np.array([ranges[n][1] for n in feat_names])
    span = hi - lo

    if A is None:
        A = np.full(G, 0.7)
    else:
        A = np.asarray(A, dtype=float)
        assert len(A) == G

    centers = sample_centers(ranges, G, min_sep_ratio=min_sep_ratio, seed=seed)
    sigma = sigma_ratio  # 정규화 거리 기준이므로 스칼라 하나로 충분 (영역별로 다르게 하려면 배열화)

    # 관측치 균등 무작위 생성 (콘크리트 실제 범위 안에서)
    X = rng.uniform(lo, hi, size=(N, P))

    # 정규화 거리 기반 가우시안 언덕들의 합
    D = np.full(N, baseline)
    dist_to_centers = np.zeros((N, G))
    for g in range(G):
        d_norm = np.sqrt(np.mean(((X - centers[g]) / span) ** 2, axis=1))
        dist_to_centers[:, g] = d_norm
        D += A[g] * np.exp(-(d_norm ** 2) / (2 * sigma ** 2))

    D += rng.normal(0, noise_std, size=N)
    D = np.clip(D, 0.0, 1.0)

    membership = np.argmin(dist_to_centers, axis=1)  # 참고용(가장 가까운 중심)

    return dict(X=X, D=D, centers=centers, A=A, sigma=sigma,
               feat_names=feat_names, membership=membership,
               dist_to_centers=dist_to_centers)


# ============================================= 간단 확인용 실행부
if __name__ == '__main__':
    # 콘크리트 실제 CSV가 있으면 그 범위를 사용, 없으면 기본값(CONCRETE_RANGES) 사용
    import os
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'concrete_dataset.csv')
    ranges = load_real_ranges(csv_path) if os.path.exists(csv_path) else CONCRETE_RANGES

    data = generate_synthetic_data(ranges, N=1000, G=3, A=[0.9, 0.7, 0.5],
                                   sigma_ratio=0.15, noise_std=0.05, seed=0)

    print(f'생성된 관측치: {data["X"].shape}')
    print(f'정답 중심 {len(data["centers"])}개:')
    for g, c in enumerate(data['centers']):
        cnt = (data['membership'] == g).sum()
        print(f'  영역 {g} (A={data["A"][g]}): 가장 가까운 관측치 {cnt}개, '
              f'중심 desirability(노이즈 전 이론값)={data["A"][g]+0.05:.2f}')
    print(f'\nD 분포: min={data["D"].min():.3f}, max={data["D"].max():.3f}, '
          f'mean={data["D"].mean():.3f}')
