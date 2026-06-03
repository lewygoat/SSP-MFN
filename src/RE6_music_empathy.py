"""RE6 · 音乐经历 ↔ 共情维度真实回归 (Training-Emotions n=263 → Study1 122 + Study2 133)

输入  real_demographic_subjects.parquet
模型  OLS / Spearman + 分组 ANOVA (Mt vs Mus vs Psy vs nomus)
输出  实验/results/RE6_music_empathy.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
OUT = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n": len(x), "error": "too_few"}
    res = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "r2": float(res.rvalue ** 2),
        "p": float(res.pvalue),
        "stderr": float(res.stderr),
    }


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n": len(x), "error": "too_few"}
    rho, p = stats.spearmanr(x, y)
    return {"n": int(len(x)), "rho": float(rho), "p": float(p)}


def main():
    subj = pd.read_parquet(OUT / "real_demographic_subjects.parquet")
    s1 = subj[subj["study"] == "training_emotions_s1"].copy()
    s2 = subj[subj["study"] == "training_emotions_s2"].copy()
    print(f"[RE6] S1 n={len(s1)}, S2 n={len(s2)}")

    out = {
        "data_source": "Zenodo 17725677 Training Emotions, CC-BY-4.0",
        "n_total": int(len(subj)),
        "n_study1": int(len(s1)),
        "n_study2": int(len(s2)),
    }

    s1["age_num"] = pd.to_numeric(s1["age"], errors="coerce")
    iri_dims = {"iri_fs": "FS幻想", "iri_ec": "EC共情关怀",
                "iri_pt": "PT观点采择", "iri_pd": "PD个人痛苦"}
    s1_ols = {}
    for col, name in iri_dims.items():
        x = pd.to_numeric(s1[col], errors="coerce").to_numpy()
        s1_ols[col] = {
            "name": name,
            "by_age": ols(s1["age_num"].to_numpy(), x),
            "spearman_age": spearman(s1["age_num"].to_numpy(), x),
        }
    out["s1_iri_vs_age_ols"] = s1_ols

    group_anova = {}
    for col, name in iri_dims.items():
        groups = []
        labels = []
        for grp in ["Mt", "Mus", "Psy", "nomus"]:
            sub = pd.to_numeric(s1.loc[s1["group"] == grp, col],
                                errors="coerce").dropna().to_numpy()
            if len(sub) >= 2:
                groups.append(sub)
                labels.append(grp)
        f, p = stats.f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)
        descs = {
            l: {"n": int(len(g)), "mean": round(float(g.mean()), 3),
                "std": round(float(g.std(ddof=1)), 3)}
            for l, g in zip(labels, groups)
        }
        pairs = {}
        if "Mus" in descs and "nomus" in descs:
            g1 = groups[labels.index("Mus")]
            g2 = groups[labels.index("nomus")]
            t, pp = stats.ttest_ind(g1, g2, equal_var=False)
            pairs["Mus_vs_nomus"] = {
                "t": float(t), "p": float(pp),
                "diff_mean": round(float(g1.mean() - g2.mean()), 3),
                "cohens_d": round(float(
                    (g1.mean() - g2.mean()) / np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                ), 3) if (g1.var(ddof=1) + g2.var(ddof=1)) > 0 else np.nan,
            }
        group_anova[col] = {
            "name": name,
            "anova_F": float(f) if np.isfinite(f) else None,
            "anova_p": float(p) if np.isfinite(p) else None,
            "group_desc": descs,
            "pairwise": pairs,
        }
    out["s1_iri_by_group_anova"] = group_anova

    s2["years"] = pd.to_numeric(s2["music_experience_years"], errors="coerce")
    s2["age_num"] = pd.to_numeric(s2["age"], errors="coerce")
    out["s2_correlations"] = {
        "years_vs_msceit_qtot": {
            "ols": ols(s2["years"].to_numpy(),
                       pd.to_numeric(s2["msceit_qtot"], errors="coerce").to_numpy()),
            "spearman": spearman(s2["years"].to_numpy(),
                                  pd.to_numeric(s2["msceit_qtot"], errors="coerce").to_numpy()),
        },
        "years_vs_met_total": {
            "ols": ols(s2["years"].to_numpy(),
                       pd.to_numeric(s2["met_total"], errors="coerce").to_numpy()),
            "spearman": spearman(s2["years"].to_numpy(),
                                  pd.to_numeric(s2["met_total"], errors="coerce").to_numpy()),
        },
        "age_vs_met_total": {
            "ols": ols(s2["age_num"].to_numpy(),
                       pd.to_numeric(s2["met_total"], errors="coerce").to_numpy()),
        },
    }

    (RES / "RE6_music_empathy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float)
    )

    print(f"  S1 IRI 按音乐组群比较 (Mus vs nomus):")
    for col in iri_dims:
        pr = group_anova[col]["pairwise"].get("Mus_vs_nomus")
        if pr:
            print(f"    {col:<8} diff={pr['diff_mean']:+.2f} p={pr['p']:.4f} d={pr['cohens_d']}")
    print(f"  S2 乐器年限 ↔ MSCEIT:")
    cc = out['s2_correlations']['years_vs_msceit_qtot']['ols']
    print(f"    OLS β={cc.get('slope', 0):.3f} R²={cc.get('r2', 0):.4f} p={cc.get('p', 1):.4f} n={cc.get('n', 0)}")
    print(f"saved → {RES / 'RE6_music_empathy.json'}")


if __name__ == "__main__":
    main()
