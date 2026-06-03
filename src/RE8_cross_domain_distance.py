"""RE8 · 跨域距离: 基于 MERT 嵌入的 corpus / ethnic 域差异度量

输入  features/RE1_mert_embeddings.npz (1950 切片 × 768)
度量  MMD (RBF) + 中心距离 + cosine 离散度 + KL 估计
任务  量化不同 corpus / ethnic 之间的特征分布距离, 评估民族音乐域差异
输出  实验/results/RE8_cross_domain_distance.json
      实验/results/RE8_corpus_distance_matrix.csv
      实验/results/RE8_ethnic_distance_matrix.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
FEAT = ROOT / "实验" / "features"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)


def mmd_rbf(X, Y, sigma=None) -> float:
    if sigma is None:
        D = np.concatenate([X, Y])
        d = np.linalg.norm(D[:, None] - D[None, :], axis=2)
        sigma = float(np.median(d[d > 0]))
    gamma = 1.0 / (2 * sigma ** 2)
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def center_distance(X, Y) -> float:
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def cosine_spread_within(X) -> float:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    n = len(X)
    iu = np.triu_indices(n, k=1)
    return float(np.median(1.0 - S[iu]))


def main():
    z = np.load(FEAT / "RE1_mert_embeddings.npz", allow_pickle=True)
    E = z["embeddings"]
    corpus = z["corpus"]
    ethnic = z["ethnic_group"]
    print(f"[RE8] N = {len(E)} embeddings × {E.shape[1]} dim")

    rng = np.random.default_rng(2026)
    sample_size = 80

    def sample_group(group_labels):
        return {
            g: rng.choice(
                np.where(group_labels == g)[0],
                size=min((group_labels == g).sum(), sample_size),
                replace=False,
            )
            for g in np.unique(group_labels)
        }

    corp_idx = sample_group(corpus)
    eth_idx = sample_group(ethnic)

    print("  computing corpus×corpus MMD/center distance...")
    corpora = list(corp_idx.keys())
    n_c = len(corpora)
    mmd_mat = np.zeros((n_c, n_c))
    cd_mat = np.zeros((n_c, n_c))
    for i, ci in enumerate(corpora):
        for j, cj in enumerate(corpora):
            Xi = E[corp_idx[ci]]
            Xj = E[corp_idx[cj]]
            mmd_mat[i, j] = mmd_rbf(Xi, Xj)
            cd_mat[i, j] = center_distance(Xi, Xj)

    corp_mmd = pd.DataFrame(mmd_mat, index=corpora, columns=corpora).round(4)
    corp_cd = pd.DataFrame(cd_mat, index=corpora, columns=corpora).round(4)
    corp_mmd.to_csv(RES / "RE8_corpus_mmd_matrix.csv")
    corp_cd.to_csv(RES / "RE8_corpus_center_distance.csv")

    print("  computing ethnic×ethnic MMD/center distance...")
    eths = list(eth_idx.keys())
    n_e = len(eths)
    e_mmd = np.zeros((n_e, n_e))
    e_cd = np.zeros((n_e, n_e))
    for i, ei in enumerate(eths):
        for j, ej in enumerate(eths):
            Xi = E[eth_idx[ei]]
            Xj = E[eth_idx[ej]]
            e_mmd[i, j] = mmd_rbf(Xi, Xj)
            e_cd[i, j] = center_distance(Xi, Xj)

    eth_mmd = pd.DataFrame(e_mmd, index=eths, columns=eths).round(4)
    eth_cd = pd.DataFrame(e_cd, index=eths, columns=eths).round(4)
    eth_mmd.to_csv(RES / "RE8_ethnic_mmd_matrix.csv")
    eth_cd.to_csv(RES / "RE8_ethnic_center_distance.csv")

    spread = {
        "corpus": {c: round(cosine_spread_within(E[corp_idx[c]]), 4) for c in corpora},
        "ethnic": {e: round(cosine_spread_within(E[eth_idx[e]]), 4) for e in eths},
    }

    minor = {"dong", "tibetan", "mongolian", "bai"}
    minor_present = [e for e in eths if e in minor]
    if "han_chinese" in eths and minor_present:
        han_idx = eth_idx["han_chinese"]
        cn_minority_dist = {}
        for m in minor_present:
            d = mmd_rbf(E[han_idx], E[eth_idx[m]])
            cn_minority_dist[m] = round(float(d), 4)
        han_minor = {"min_ethnic_distance_mean": float(round(np.mean(list(cn_minority_dist.values())), 4)),
                     "per_minority": cn_minority_dist}
    else:
        han_minor = None

    if "non-ethnic_western_choral" in eths and minor_present:
        west_idx = eth_idx["non-ethnic_western_choral"]
        cn_west = {}
        for m in minor_present:
            d = mmd_rbf(E[west_idx], E[eth_idx[m]])
            cn_west[m] = round(float(d), 4)
        west_minor = {"west_to_minority_mean": float(round(np.mean(list(cn_west.values())), 4)),
                      "per_minority": cn_west}
    else:
        west_minor = None

    summary = {
        "input": "MERT-v1-95M 768-D embeddings from 1950 real 30s clips",
        "metric": "MMD (RBF, sigma=median pairwise) + Euclidean center distance",
        "sample_per_group": sample_size,
        "corpus_distance_summary": {
            "max_mmd_pair": {
                "pair": [corpora[i], corpora[j]]
                for i in range(n_c) for j in range(i + 1, n_c)
                if mmd_mat[i, j] == mmd_mat[np.triu_indices(n_c, k=1)].max()
            } if n_c >= 2 else None,
            "mean_offdiag_mmd": float(round(mmd_mat[np.triu_indices(n_c, k=1)].mean(), 4)),
        },
        "ethnic_distance_summary": {
            "mean_offdiag_mmd": float(round(e_mmd[np.triu_indices(n_e, k=1)].mean(), 4)),
        },
        "within_group_cosine_spread": spread,
        "han_to_minority_domain_gap": han_minor,
        "west_to_minority_domain_gap": west_minor,
        "corpora_present": corpora,
        "ethnic_groups_present": eths,
    }
    (RES / "RE8_cross_domain_distance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print(f"\n  Corpus 间 MMD 矩阵:\n{corp_mmd}")
    print(f"\n  Ethnic 间 MMD 矩阵:\n{eth_mmd}")
    if han_minor:
        print(f"\n  han_chinese → 少数民族 MMD: {han_minor['per_minority']}")
    if west_minor:
        print(f"\n  Western choral → 少数民族 MMD: {west_minor['per_minority']}")
    print(f"\nsaved → {RES / 'RE8_cross_domain_distance.json'}")


if __name__ == "__main__":
    main()
