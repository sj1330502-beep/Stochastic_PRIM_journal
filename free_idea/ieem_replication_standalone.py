"""
================================================================================
IEEM 원논문 방법론 재현 (2단계 포함 최종 버전)
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 설정
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
ALPHA_PEEL = 0.05
MIN_SUPPORT = 100
S_OPTIONS = (4, 5, 6)
T_PER_SIZE = 667
SEED_PRIM = 1
MC_SAMPLES = 10000

# ---------------- 함수
def desirability_ntb(y):
    # D 계산 (nominal-the-better)
    d = np.zeros_like(y, dtype=float)
    lo = (y >= 20.0) & (y <= 60.0)
    d[lo] = (y[lo] - 20.0) / (60.0 - 20.0)
    hi = (y > 60.0) & (y <= 100.0)
    d[hi] = (100.0 - y[hi]) / (100.0 - 60.0)
    return d

def peel_trajectory(X, D, S, rng):
    P = X.shape[1]
    idx = np.arange(len(D))
    best_idx, best_obj = idx.copy(), D.mean()
    while True:
        if len(idx) * (1 - ALPHA_PEEL) < MIN_SUPPORT: break
        feats = rng.choice(P, size=S, replace=False)
        cand_keep, cand_obj = None, -np.inf
        for p in feats:
            xp = X[idx, p]
            for keep in (idx[xp > np.quantile(xp, ALPHA_PEEL)], idx[xp < np.quantile(xp, 1-ALPHA_PEEL)]):
                if len(keep) >= MIN_SUPPORT:
                    obj = D[keep].mean()
                    if obj > cand_obj: cand_obj, cand_keep = obj, keep
        if cand_keep is None: break
        idx = cand_keep
        if cand_obj > best_obj: best_obj, best_idx = cand_obj, idx.copy()
    return best_idx, best_obj

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")

    df = pd.read_csv(CSV_PATH)
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    
    # 1. 모델 학습 및 박스 생성
    model = RandomForestRegressor(n_estimators=100, random_state=SEED_PRIM).fit(X, y)
    D = desirability_ntb(y)
    
    boxes = []
    rng = np.random.default_rng(SEED_PRIM)
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append(dict(idx=idx, support=len(idx), dbar=obj))
    
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    
    # 2. 클러스터링
    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)
    sim = inter / (sup[:, None] + sup[None, :] - inter)
    dist = np.clip(1.0 - sim, 0, None)
    Z = linkage(squareform(dist, checks=False), method='average')

    # 3. 통합 박스 생성 및 2단계 평가
    lab = fcluster(Z, int(M * 0.1), criterion='maxclust')
    print("\n[2단계] 결과 분석:")
    for cluster_id in np.unique(lab):
        member_idx = np.where(lab == cluster_id)[0]
        D_orig = np.mean([boxes[i]['dbar'] for i in member_idx])
        
        # 통합 박스 경계
        all_idx = np.concatenate([boxes[i]['idx'] for i in member_idx])
        g_min, g_max = X[all_idx].min(axis=0), X[all_idx].max(axis=0)
        
        # 몬테카를로 평가
        samples = np.random.uniform(g_min, g_max, size=(MC_SAMPLES, X.shape[1]))
        sample_D = desirability_ntb(model.predict(samples))
        
        delta_D = D_orig - np.percentile(sample_D, 5)
        print(f"Cluster {cluster_id}: D_orig={D_orig:.4f}, Delta_D={delta_D:.4f}")

if __name__ == '__main__':
    main()