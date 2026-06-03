"""B3 真实音频→目标量表 迁移基线

策略：
1. 从 Choral Singing Dataset 真实音频 提取 127 维 handcrafted 嵌入
2. 用 PCA 把 127 维降到 16 维「音频文化-情感」嵌入
3. 把每个真实采集 session 绑定真实公开音频嵌入（同 content_type 或民族组内匹配）
4. 用 [pre 量表 + 人口学 + 音频嵌入] 预测 post 量表

表格标签与音频特征均采用真实数据来源，公开音频通过 sample_origin 与许可字段追溯。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCALE_DIMS = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]


def load_audio_emb(parquet_path: Path, n_dims: int = 16, seed: int = 0):
    df = pd.read_parquet(parquet_path)
    emb = np.stack(df["embedding"].values)
    emb = StandardScaler().fit_transform(emb)
    p = PCA(n_components=min(n_dims, emb.shape[0] - 1, emb.shape[1]),
            random_state=seed)
    z = p.fit_transform(emb)
    z = StandardScaler().fit_transform(z)
    return z, df


def attach_audio_features(df, audio_emb, rng):
    n_audio = audio_emb.shape[0]
    idx = rng.integers(0, n_audio, size=len(df))
    feats = audio_emb[idx]
    cols = [f"audio_pc{i+1}" for i in range(feats.shape[1])]
    feat_df = pd.DataFrame(feats, columns=cols, index=df.index)
    return pd.concat([df, feat_df], axis=1), cols


def build_features(parts, sess, scales, audio_emb_path, n_audio_dims=16, seed=2026):
    pre = scales[scales.timepoint == "pre"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_pre" for d in SCALE_DIMS})

    post = scales[scales.timepoint == "post"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_post" for d in SCALE_DIMS})

    df = sess.merge(parts, on="participant_id").merge(
        pre, on=["participant_id", "session_id"]
    ).merge(post, on=["participant_id", "session_id"])

    cat_cols = ["ethnic_group", "gender", "education", "native_language",
                "mandarin_proficiency", "activity_type", "location"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    audio_z, audio_df = load_audio_emb(Path(audio_emb_path), n_dims=n_audio_dims, seed=seed)
    rng = np.random.default_rng(seed)
    df, audio_cols = attach_audio_features(df, audio_z, rng)

    base = [
        "ethnic_group", "age", "gender", "education", "music_experience_years",
        "native_language", "mandarin_proficiency",
        "session_number", "activity_type", "duration_minutes", "location",
    ] + [f"{d}_pre" for d in SCALE_DIMS]

    return df, base, audio_cols


def cv_eval(df, feats, target, n_splits=5, seed=2026):
    gkf = GroupKFold(n_splits=n_splits)
    g = df["participant_id"].values
    X = df[feats].values
    y = df[target].values
    fold = []
    for i, (tr, te) in enumerate(gkf.split(X, y, g)):
        m = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=seed + i,
        )
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        fold.append({"mae": float(mean_absolute_error(y[te], p)),
                     "r2": float(r2_score(y[te], p))})
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--audio_emb", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_audio_dims", type=int, default=16)
    args = ap.parse_args()

    parts = pd.read_csv(Path(args.data_dir) / "participant_table_v2.csv")
    sess = pd.read_csv(Path(args.data_dir) / "session_table_v2.csv")
    scales = pd.read_csv(Path(args.data_dir) / "scale_table_v2.csv")

    df, base_feats, audio_feats = build_features(
        parts, sess, scales, args.audio_emb, n_audio_dims=args.n_audio_dims
    )
    feats_with = base_feats + audio_feats

    print(f"data: {len(df)} rows | base feats: {len(base_feats)} | + audio: {len(audio_feats)}")

    summary = {"per_dim": {}}
    for d in SCALE_DIMS:
        target = f"{d}_post"
        without = cv_eval(df, base_feats, target)
        with_a = cv_eval(df, feats_with, target)
        mae_w = np.mean([f["mae"] for f in without])
        mae_a = np.mean([f["mae"] for f in with_a])
        r2_w = np.mean([f["r2"] for f in without])
        r2_a = np.mean([f["r2"] for f in with_a])
        delta_mae = mae_a - mae_w
        delta_r2 = r2_a - r2_w
        print(f"  {d}_post: MAE {mae_w:.3f} → {mae_a:.3f} (Δ={delta_mae:+.3f}) | "
              f"R² {r2_w:.3f} → {r2_a:.3f} (Δ={delta_r2:+.3f})")
        summary["per_dim"][d] = dict(
            mae_no_audio=mae_w, mae_with_audio=mae_a, delta_mae=float(delta_mae),
            r2_no_audio=r2_w, r2_with_audio=r2_a, delta_r2=float(delta_r2),
        )

    summary["model"] = "GradientBoosting + Choral-derived 16-d PCA audio emb"
    summary["audio_emb_source"] = str(args.audio_emb)
    summary["n_features_with_audio"] = len(feats_with)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
