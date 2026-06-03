"""B2 文本（讨论/歌词）→ 量表 基线

真实数据 v2 暂未生成讨论/歌词文本（仅 ID）。本脚本支持两种模式：
1. --use_old_text 用旧版 v1 的 discussion_table.csv / lyric_table.csv 文本
2. 否则生成基于 IRI 题项语义的 stub 文本（用于跑通管线）

文本嵌入 → 256 维向量（TF-IDF char n-gram 回退保证零外网依赖）
然后 GBR 预测 post 量表
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

SCALE_DIMS = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]


def encode_texts_tfidf_svd(texts, n_dims=64, seed=2026):
    vec = TfidfVectorizer(
        max_features=4000, analyzer="char_wb", ngram_range=(2, 4),
        min_df=2, sublinear_tf=True,
    )
    M = vec.fit_transform(texts)
    n_dims = min(n_dims, max(2, M.shape[1] - 1), M.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_dims, random_state=seed)
    return svd.fit_transform(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--text_csv", default=None,
                    help="旧版 discussion_table.csv 路径；不提供则用 stub")
    ap.add_argument("--text_col", default="discussion_text")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_text_dims", type=int, default=32)
    args = ap.parse_args()

    parts = pd.read_csv(Path(args.data_dir) / "participant_table_v2.csv")
    sess = pd.read_csv(Path(args.data_dir) / "session_table_v2.csv")
    scales = pd.read_csv(Path(args.data_dir) / "scale_table_v2.csv")

    pre = scales[scales.timepoint == "pre"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_pre" for d in SCALE_DIMS})
    post = scales[scales.timepoint == "post"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALE_DIMS]
    ].rename(columns={f"{d.lower()}_total": f"{d}_post" for d in SCALE_DIMS})

    df = sess.merge(parts, on="participant_id").merge(
        pre, on=["participant_id", "session_id"]
    ).merge(post, on=["participant_id", "session_id"])

    if args.text_csv and Path(args.text_csv).exists():
        text_df = pd.read_csv(args.text_csv)
        if args.text_col not in text_df.columns:
            txt_cols = [c for c in text_df.columns if "text" in c.lower() or "content" in c.lower() or "lyric" in c.lower()]
            args.text_col = txt_cols[0] if txt_cols else text_df.columns[-1]
            print(f"using text column: {args.text_col}")
        if "session_id" in text_df.columns:
            df = df.merge(text_df[["session_id", args.text_col]], on="session_id", how="left")
        else:
            df[args.text_col] = text_df[args.text_col].iloc[: len(df)].values
        texts = df[args.text_col].fillna("空").astype(str).tolist()
        text_source = f"loaded from {args.text_csv}"
    else:
        rng = np.random.default_rng(2026)
        stub_words = ["共情", "理解", "尊重", "倾听", "情感", "互动", "民族", "音乐",
                      "歌曲", "讨论", "分享", "学习", "文化", "传统", "节奏", "旋律",
                      "深刻", "温暖", "感动", "新颖"]
        texts = []
        for _ in range(len(df)):
            n = int(rng.integers(8, 25))
            texts.append("，".join(rng.choice(stub_words, size=n).tolist()))
        text_source = "stub (random word bag)"

    text_emb = encode_texts_tfidf_svd(texts, n_dims=args.n_text_dims)
    text_cols = [f"txt_{i+1}" for i in range(text_emb.shape[1])]
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(text_emb, columns=text_cols)], axis=1)

    cat_cols = ["ethnic_group", "gender", "education", "native_language",
                "mandarin_proficiency", "activity_type", "location"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    base_feats = [
        "ethnic_group", "age", "gender", "education", "music_experience_years",
        "native_language", "mandarin_proficiency",
        "session_number", "activity_type", "duration_minutes", "location",
    ] + [f"{d}_pre" for d in SCALE_DIMS]
    feats_with_text = base_feats + text_cols

    print(f"data: {len(df)} | base: {len(base_feats)} | text dims: {len(text_cols)} | source: {text_source}")

    summary = {"per_dim": {}, "text_source": text_source, "n_text_dims": text_emb.shape[1]}
    gkf = GroupKFold(n_splits=5)
    for d in SCALE_DIMS:
        target = f"{d}_post"
        for tag, feats in [("no_text", base_feats), ("with_text", feats_with_text)]:
            X = df[feats].values
            y = df[target].values
            g = df["participant_id"].values
            maes, r2s = [], []
            for i, (tr, te) in enumerate(gkf.split(X, y, g)):
                m = GradientBoostingRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.85, random_state=2026 + i,
                )
                m.fit(X[tr], y[tr])
                p = m.predict(X[te])
                maes.append(mean_absolute_error(y[te], p))
                r2s.append(r2_score(y[te], p))
            summary["per_dim"].setdefault(d, {})[tag] = {
                "mae_mean": float(np.mean(maes)),
                "r2_mean": float(np.mean(r2s)),
            }
        s = summary["per_dim"][d]
        print(f"  {d}_post: MAE {s['no_text']['mae_mean']:.3f} → {s['with_text']['mae_mean']:.3f} | "
              f"R² {s['no_text']['r2_mean']:.3f} → {s['with_text']['r2_mean']:.3f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
