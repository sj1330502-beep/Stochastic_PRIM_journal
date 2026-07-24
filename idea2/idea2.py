import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# --- 1. 설정 및 데이터 로드 ---
ALPHA_TV, BETA_TV = 0.9, 0.1  # Tversky 비대칭 가중치 (차별화를 위해 0.9, 0.1로 조정 권장)
DATA_PATH = 'concrete_dataset.csv'

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    target_col = [c for c in df.columns if 'strength' in c.lower()][0]
    df = df.rename(columns={target_col: 'Strength'})
    return df.drop('Strength', axis=1), df['Strength']

# --- 2. 원논문 기반 박스 생성 ---
def build_boxes(X, n_subsets=15):
    boxes = []
    # PRIM 알고리즘 모사: Cement(첫번째 변수) 기준 박스 샘플링
    col = X.columns[0]
    for i in range(n_subsets):
        start, end = X[col].quantile(i/n_subsets), X[col].quantile((i+1)/n_subsets)
        idx = np.where((X[col] >= start) & (X[col] <= end))[0]
        if len(idx) > 0:
            boxes.append({'idx': idx, 'support': len(idx)})
    return boxes

# --- 3. 유사도 및 클러스터링 ---
def get_linkage(boxes, method='jaccard'):
    M = len(boxes)
    sets = [set(b['idx']) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    
    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)
    
    if method == 'jaccard':
        denom = sup[:, None] + sup[None, :] - inter
    else: # Tversky
        denom = inter + ALPHA_TV * (sup[:, None] - inter) + BETA_TV * (sup[None, :] - inter)
        
    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom!=0)
    sim = (sim + sim.T) / 2
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average'), boxes

# --- 4. 효율성 검증 (ECC 계산) ---
def calculate_ecc(Z, boxes, X, y):
    labels = fcluster(Z, 3, criterion='maxclust')
    ecc_scores = []
    for i in set(labels):
        cluster_indices = np.concatenate([boxes[j]['idx'] for j in np.where(labels == i)[0]])
        # 정밀도(부피 역수)와 품질(평균 강도) 반영
        volume = np.prod([X.loc[cluster_indices, col].max() - X.loc[cluster_indices, col].min() + 1e-6 for col in X.columns])
        mean_strength = y.loc[cluster_indices].mean()
        ecc_scores.append(mean_strength / (volume**(1/len(X.columns))))
    return np.mean(ecc_scores)

# --- 5. 메인 실행 ---
def main():
    X, y = load_data()
    boxes = build_boxes(X)
    
    for method in ['jaccard', 'tversky']:
        Z, _ = get_linkage(boxes, method=method)
        ecc = calculate_ecc(Z, boxes, X, y)
        print(f"[{method.upper()} 방식 성능 지표]")
        print(f" -> ECC Score: {ecc:.4f}")
    
    print("\n[분석] Tversky 지표가 Jaccard보다 높다면 제안 아이디어가 정밀 수축에 더 유리함을 입증함.")

if __name__ == '__main__':
    main()