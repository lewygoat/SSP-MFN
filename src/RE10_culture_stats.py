"""RE10 · 真实文化元数据全表统计 + 卡方独立性检验

输入  real_culture_metadata_table.parquet (781 条 × 7 corpus × 7 ethnic × 31 genre × 185 artist × 18 region)
输出  实验/results/RE10_culture_stats.json
      实验/results/RE10_crosstab_*.csv
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


def chi2(table: pd.DataFrame) -> dict:
    vals = table.values
    if vals.shape[0] < 2 or vals.shape[1] < 2:
        return {"error": "single_dimension"}
    if vals.sum() < 5:
        return {"error": "too_few"}
    try:
        chi, p, dof, _ = stats.chi2_contingency(vals)
        n = vals.sum()
        cramers_v = float(np.sqrt(chi / (n * (min(vals.shape) - 1))))
        return {"chi2": float(chi), "p": float(p), "dof": int(dof),
                "n": int(n), "cramers_v": round(cramers_v, 4)}
    except Exception as e:
        return {"error": str(e)}


def main():
    df = pd.read_parquet(OUT / "real_culture_metadata_table.parquet")
    print(f"[RE10] n records = {len(df)}")

    desc = {
        "n_records": int(len(df)),
        "n_unique": {
            c: int(df[c].nunique())
            for c in ["corpus", "country", "ethnic_group", "language", "genre",
                      "region", "artist", "ritual_function"]
        },
    }

    ct1 = pd.crosstab(df["corpus"], df["ethnic_group"])
    ct1.to_csv(RES / "RE10_crosstab_corpus_ethnic.csv")

    ct2 = pd.crosstab(df["ethnic_group"], df["language"])
    ct2.to_csv(RES / "RE10_crosstab_ethnic_language.csv")

    ct3 = pd.crosstab(df["genre"], df["ritual_function"])
    ct3.to_csv(RES / "RE10_crosstab_genre_ritual.csv")

    ct4 = pd.crosstab(df["country"], df["genre"])
    ct4.to_csv(RES / "RE10_crosstab_country_genre.csv")

    ct5 = pd.crosstab(df["ethnic_group"], df["artist_gender"])
    ct5.to_csv(RES / "RE10_crosstab_ethnic_gender.csv")

    ind = df[df["corpus"] == "IndianFolkMusic"].copy()
    indian_stats = {}
    if not ind.empty:
        indian_stats["n"] = int(len(ind))
        indian_stats["n_genres"] = int(ind["genre"].nunique())
        indian_stats["n_artists"] = int(ind["artist"].nunique())
        indian_stats["genre_top10"] = (
            ind["genre"].value_counts().head(10).to_dict()
        )
        indian_stats["gender_dist"] = ind["artist_gender"].value_counts().to_dict()
        indian_stats["state_dist"] = ind["region"].value_counts().to_dict()

        ind["n_artists_in_recording"] = pd.to_numeric(
            ind["n_artists_in_recording"], errors="coerce"
        )
        indian_stats["artists_per_recording"] = {
            "mean": float(round(ind["n_artists_in_recording"].mean(), 3)),
            "max": int(ind["n_artists_in_recording"].max()),
            "p90": float(ind["n_artists_in_recording"].quantile(0.9)),
        }

        ct_genre_state = pd.crosstab(ind["genre"], ind["region"])
        ct_genre_state.to_csv(RES / "RE10_crosstab_indian_genre_state.csv")
        indian_stats["chi2_genre_state"] = chi2(ct_genre_state)

        ct_genre_gender = pd.crosstab(ind["genre"], ind["artist_gender"])
        indian_stats["chi2_genre_gender"] = chi2(ct_genre_gender)

    chi2_summary = {
        "corpus_x_ethnic": chi2(ct1),
        "ethnic_x_language": chi2(ct2),
        "genre_x_ritual": chi2(ct3),
        "country_x_genre": chi2(ct4),
        "ethnic_x_gender": chi2(ct5),
    }

    result = {
        "descriptive": desc,
        "chi2_independence_tests": chi2_summary,
        "indian_folk_zoomed": indian_stats,
        "license_distribution": df["license"].value_counts().to_dict(),
        "data_sources": df.groupby("corpus")["doi"].first().to_dict(),
    }
    (RES / "RE10_culture_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float)
    )

    print(f"  unique: {desc['n_unique']}")
    print(f"  χ²(corpus×ethnic): {chi2_summary['corpus_x_ethnic']}")
    print(f"  χ²(genre×ritual): {chi2_summary['genre_x_ritual']}")
    if indian_stats:
        print(f"  Indian Folk: {indian_stats['n']} 录音 × {indian_stats['n_genres']} 风格 × {indian_stats['n_artists']} 艺人")
        print(f"  χ²(genre×state): {indian_stats['chi2_genre_state']}")
    print(f"saved → {RES / 'RE10_culture_stats.json'}")


if __name__ == "__main__":
    main()
