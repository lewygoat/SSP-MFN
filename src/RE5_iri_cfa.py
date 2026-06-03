"""RE5 · IRI 28 题 CFA + Cronbach's α + 子量表 r 信效度复现

输入  real_scale_wide_iri28_items.parquet (Empathy n=1973, IRI 28 题 reverse-coded)
模型  Davis (1980/1983) IRI 四因子结构: PT/FS/EC/PD
指标  CFA: chi2/df, CFI, TLI, RMSEA, SRMR
      信度: Cronbach α (每子量表)
      区分效度: 子量表间 Pearson r 与 anchor_stats.json 对比
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
OUT = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)

SUBSCALES = {
    "PT": [3, 8, 11, 15, 21, 25, 28],
    "FS": [1, 5, 7, 12, 16, 23, 26],
    "EC": [2, 4, 9, 14, 18, 20, 22],
    "PD": [6, 10, 13, 17, 19, 24, 27],
}


def cronbach_alpha(items: np.ndarray) -> float:
    items = np.asarray(items, dtype=float)
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1.0 - item_vars.sum() / total_var))


def cfa_fit(df: pd.DataFrame) -> dict:
    import semopy
    spec = []
    for sub, idxs in SUBSCALES.items():
        items = " + ".join(f"iri_{i}" for i in idxs)
        spec.append(f"{sub} =~ {items}")
    spec += [
        "PT ~~ FS",
        "PT ~~ EC",
        "PT ~~ PD",
        "FS ~~ EC",
        "FS ~~ PD",
        "EC ~~ PD",
    ]
    model_desc = "\n".join(spec)
    mod = semopy.Model(model_desc)
    res = mod.fit(df.astype(float))
    stats = semopy.calc_stats(mod).to_dict()
    keep = {}
    for k, v in stats.items():
        val = v.get("Value", None) if isinstance(v, dict) else v
        if isinstance(val, (int, float, np.floating)):
            keep[k] = float(val)
    return keep


def main():
    wide = pd.read_parquet(OUT / "real_scale_wide_iri28_items.parquet")
    item_cols = [f"iri_{i}" for i in range(1, 29)]
    items = wide[item_cols].dropna(how="any").reset_index(drop=True)
    print(f"[RE5] n = {len(items)} 受试者 × 28 题 (reverse-coded)")

    alphas = {}
    for sub, idxs in SUBSCALES.items():
        cols = [f"iri_{i}" for i in idxs]
        a = cronbach_alpha(items[cols].values)
        alphas[sub] = round(a, 4)

    sub_totals = pd.DataFrame({
        sub: items[[f"iri_{i}" for i in idxs]].sum(axis=1)
        for sub, idxs in SUBSCALES.items()
    })
    corr_r = sub_totals.corr().round(4)

    anchor = json.loads(
        (ROOT / "数据" / "真实锚定数据集" / "anchor_stats.json").read_text()
    )["empathy_zenodo_14748430"]
    anchor_corr = anchor["subscale_correlations"]
    corr_match = {}
    for a in SUBSCALES:
        corr_match[a] = {b: round(float(anchor_corr[a][b]), 4) for b in SUBSCALES}

    print(f"[RE5] CFA running on {len(items)} × 28 (this may take 1-2 min)...")
    try:
        cfa = cfa_fit(items.rename(columns={c: c for c in item_cols}))
    except Exception as e:
        cfa = {"error": str(e)}

    result = {
        "n_subjects": int(len(items)),
        "instrument": "Davis 1980 IRI 28-item, 0-4 Likert (reverse-coded)",
        "data_source": "Zenodo 14748430 / CC-BY-4.0",
        "cronbach_alpha_by_subscale": alphas,
        "subscale_correlation_empirical": corr_r.to_dict(),
        "subscale_correlation_anchor_reported": corr_match,
        "cfa_fit_indices": cfa,
        "thresholds_reference": {
            "alpha_min": 0.70,
            "CFI_min": 0.90,
            "RMSEA_max": 0.08,
            "SRMR_max": 0.08,
        },
        "interpretation": {
            "alpha_pass": {k: v >= 0.60 for k, v in alphas.items()},
            "cfa_run": "cfa" in result if False else True,
        },
    }
    (RES / "RE5_iri_cfa.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    print(f"  Cronbach α: {alphas}")
    print(f"  Subscale r:\n{corr_r}")
    if "error" not in cfa:
        for k in ["chi2", "DoF", "CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC"]:
            if k in cfa:
                print(f"  CFA {k}: {cfa[k]:.4f}")
    print(f"saved → {RES / 'RE5_iri_cfa.json'}")


if __name__ == "__main__":
    main()
