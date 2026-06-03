"""B1 LightGBM/GBR 表格基线 — 用真实数据预测社会技能 post 总分。

输入：真实数据 v2 的 participant + session + scale 表
输出：每位参与者每场次每维度的 post 量表预测 + 评估指标
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

SCALE_DIMS = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]


def build_features(parts, sess, scales):
    pre = scales[scales.timepoint == "pre"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_pre" for d in SCALE_DIMS})

    post = scales[scales.timepoint == "post"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_post" for d in SCALE_DIMS})

    df = sess.merge(parts, on="participant_id", how="left")
    df = df.merge(pre, on=["participant_id", "session_id"])
    df = df.merge(post, on=["participant_id", "session_id"])

    cat_cols = [
        "ethnic_group", "gender", "education", "native_language",
        "mandarin_proficiency", "activity_type", "location",
    ]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = [
        "ethnic_group", "age", "gender", "education", "music_experience_years",
        "native_language", "mandarin_proficiency",
        "session_number", "activity_type", "duration_minutes", "location",
    ] + [f"{d}_pre" for d in SCALE_DIMS]

    return df, feature_cols


def run_cv(df, feature_cols, target_col, n_splits=5, seed=2026):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["participant_id"].values
    X = df[feature_cols].values
    y = df[target_col].values

    fold_metrics = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=seed + fold,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        fold_metrics.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "mae": float(mean_absolute_error(y[te], pred)),
            "r2": float(r2_score(y[te], pred)),
            "baseline_mae_mean": float(mean_absolute_error(y[te], np.full_like(y[te], y[tr].mean(), dtype=float))),
        })
    return fold_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    parts = pd.read_csv(Path(args.data_dir) / "participant_table_v2.csv")
    sess = pd.read_csv(Path(args.data_dir) / "session_table_v2.csv")
    scales = pd.read_csv(Path(args.data_dir) / "scale_table_v2.csv")

    df, feats = build_features(parts, sess, scales)
    print(f"data: {len(df)} session-rows | features: {len(feats)}")

    results = {}
    for d in SCALE_DIMS:
        target = f"{d}_post"
        m = run_cv(df, feats, target)
        results[d] = m
        avg_mae = np.mean([f["mae"] for f in m])
        avg_r2 = np.mean([f["r2"] for f in m])
        avg_base = np.mean([f["baseline_mae_mean"] for f in m])
        print(f"  {d}_post: MAE={avg_mae:.3f} (baseline {avg_base:.3f}) | R2={avg_r2:.3f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "GradientBoostingRegressor",
        "n_splits": 5,
        "group": "participant_id",
        "n_features": len(feats),
        "feature_cols": feats,
        "per_dim": {
            d: {
                "mae_mean": float(np.mean([x["mae"] for x in results[d]])),
                "mae_std": float(np.std([x["mae"] for x in results[d]])),
                "r2_mean": float(np.mean([x["r2"] for x in results[d]])),
                "baseline_mae_mean": float(np.mean([x["baseline_mae_mean"] for x in results[d]])),
                "folds": results[d],
            } for d in SCALE_DIMS
        },
    }
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
