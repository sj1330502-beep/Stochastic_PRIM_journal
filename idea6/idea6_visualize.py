"""
================================================================================
아이디어 6번 — 목적함수 곡선 시각화
================================================================================
기존 방식(ECC 단독, 단조증가 -> 경계에서 강제 종료)과, 방식0(단순곱셈),
방식1(가중조화평균), 방식3(AIC/BIC 스타일 페널티, 신규)의 score(K) 곡선을
비교해, 어느 것이 진짜 "내부 최적점(봉우리)"을 갖는지 시각적으로 보여준다.
방식2(정규화)는 봉우리가 아니라 잡음성 결과로 판명되어 제외했다.
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'concrete_dataset.csv')
BOX_CACHE = os.path.join(HERE, 'boxes_concrete.npy')
OUT_PATH = r'C:\Users\USER\Desktop\과제\학연생\idea6\idea6_objective_curves.png'

ALPHA_PEEL, MIN_SUPPORT = 0.05, 100
S_OPTIONS, T_PER_SIZE, SEED_PRIM = (4, 5, 6), 667, 1
N_MEMB = 5
K_GRID_STEP = 5


def make_desirability(y_all):
    lower, upper = float(y_all.min()), float(y_all.max())
    def d_func(y):
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        m1 = (y >= lower) & (y <= 60.0)
        d[m1] = (y[m1] - lower) / (60.0 - lower)
        m2 = (y > 60.0) & (y <= upper)
        d[m2] = (upper - y[m2]) / (upper - 60.0)
        return np.clip(d, 0.0, 1.0)
    return d_func


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
            lo_q, hi_q = np.quantile(xp, ALPHA_PEEL), np.quantile(xp, 1 - ALPHA_PEEL)
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
    for S in S_OPTIONS:
        for _ in range(T_PER_SIZE):
            idx, obj = peel_trajectory(X, D, S, rng)
            if len(idx) >= MIN_SUPPORT:
                boxes.append({'idx': idx, 'support': len(idx), 'dbar': obj})
    return boxes


def build_linkage(boxes):
    M = len(boxes)
    sets = [set(b['idx'].tolist()) for b in boxes]
    sup = np.array([b['support'] for b in boxes], dtype=float)
    inter = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            inter[i, j] = inter[j, i] = len(sets[i] & sets[j])
    np.fill_diagonal(inter, sup)
    sim = inter / (sup[:, None] + sup[None, :] - inter)
    dist = np.clip(1.0 - sim, 0, 1.0)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='average'), sets, M


def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = df.iloc[:, :8].values.astype(float)
    y = df.iloc[:, 8].values.astype(float)
    D = make_desirability(y)(y)

    if os.path.exists(BOX_CACHE):
        boxes = list(np.load(BOX_CACHE, allow_pickle=True))
    else:
        boxes = build_boxes(X, D)
        np.save(BOX_CACHE, np.array(boxes, dtype=object), allow_pickle=True)

    Z, sets, M = build_linkage(boxes)
    k_grid = list(range(int(M * 0.05), int(M * 0.50) + 1, K_GRID_STEP))

    per_K = {}
    for K in k_grid:
        lab = fcluster(Z, K, criterion='maxclust')
        entries = []
        for k in np.unique(lab):
            mem = np.where(lab == k)[0]
            I = set.intersection(*[sets[m] for m in mem])
            U = set.union(*[sets[m] for m in mem])
            entries.append((len(mem), len(I) / len(U) if U else 0.0))
        per_K[K] = entries

    def eval_K(entries, n_memb):
        eff = [(n, g) for n, g in entries if n >= n_memb]
        if not eff:
            return None
        rho = sum(n for n, _ in eff) / M
        ecc = np.mean([g for _, g in eff])
        return ecc, rho

    vals = []
    for K in k_grid:
        res = eval_K(per_K[K], N_MEMB)
        if res is not None:
            vals.append((K,) + res)
    Ks = np.array([v[0] for v in vals])
    Es = np.array([v[1] for v in vals])
    Rs = np.array([v[2] for v in vals])

    # 각 score 곡선 계산
    w = 0.5
    score0 = Es ** w * Rs ** (1 - w)

    beta = 1.0
    score1 = (1 + beta ** 2) * Es * Rs / (beta ** 2 * Es + Rs)

    lam = 1.0
    score3 = Es - lam * (Ks / M)

    def mark_peak(ax, x, y, color, label):
        i = np.argmax(y)
        ax.plot(x, y, color=color, lw=1.8, label=label)
        ax.scatter([x[i]], [y[i]], color=color, s=70, zorder=5,
                  edgecolor='black', linewidth=1)
        ax.axvline(x[i], color=color, ls=':', lw=1, alpha=0.6)
        return x[i]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('K* 목적함수 비교: 경계 밀착(기존) vs 내부 봉우리(신규 방식)',
                 fontsize=14, fontweight='bold')

    # (0,0) 기존: ECC 단독 - 단조증가, 봉우리 없음
    ax = axes[0, 0]
    ax.plot(Ks, Es, color='gray', lw=2)
    ax.set_title('[기존] ECC(K) 단독\n→ 단조증가, 내부 봉우리 없음 (경계에서 강제 종료)')
    ax.set_xlabel('K'); ax.set_ylabel('ECC')
    ax.axhline(1.0, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.grid(alpha=0.3)

    # (0,1) 방식0: 단순곱셈
    ax = axes[0, 1]
    k0 = mark_peak(ax, Ks, score0, 'tab:blue', f'score=ECC^{w}·ρ^{1-w}')
    ax.set_title(f'[방식0] 단순 곱셈 (w={w})\n→ 내부 봉우리 K*={k0}')
    ax.set_xlabel('K'); ax.set_ylabel('score')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,0) 방식1: 조화평균
    ax = axes[1, 0]
    k1 = mark_peak(ax, Ks, score1, 'tab:green', f'F-beta(β={beta})')
    ax.set_title(f'[방식1] 가중조화평균 (β={beta})\n→ 내부 봉우리 K*={k1}')
    ax.set_xlabel('K'); ax.set_ylabel('score')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,1) 방식3: AIC/BIC 페널티 (여러 lambda 겹쳐 그리기)
    ax = axes[1, 1]
    colors = ['tab:orange', 'tab:red', 'tab:purple']
    for lam_i, color in zip([0.5, 1.0, 1.5], colors):
        s3 = Es - lam_i * (Ks / M)
        k3 = mark_peak(ax, Ks, s3, color, f'λ={lam_i}')
    ax.set_title('[방식3, 신규] AIC/BIC 페널티\nscore=ECC(K)-λ·(K/M)\n→ λ에 따라 이동하는 내부 봉우리')
    ax.set_xlabel('K'); ax.set_ylabel('score')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'저장 완료: {OUT_PATH}')
    print(f'방식0 K*={k0}, 방식1 K*={k1}')


if __name__ == '__main__':
    main()
