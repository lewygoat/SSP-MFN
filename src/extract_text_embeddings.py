"""使用预训练多语言文本编码器提取讨论/歌词文本嵌入。

默认使用 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（可中文）。
若网络不可用，自动回退到 TF-IDF 嵌入。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_text_table(path: Path, text_col_candidates):
    df = pd.read_csv(path)
    for c in text_col_candidates:
        if c in df.columns:
            return df, c
    raise ValueError(f"找不到文本列；可用列: {list(df.columns)}")


def encode_with_st(texts, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def encode_with_tfidf(texts, max_features=512):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(1, 3))
    m = vec.fit_transform(texts)
    return m.toarray().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--text_col", default=None, help="若不指定则自动推断")
    ap.add_argument("--id_col", default=None, help="主键列；不指定则用index")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ap.add_argument("--fallback_tfidf", action="store_true")
    args = ap.parse_args()

    path = Path(args.input_csv)
    candidates = [args.text_col] if args.text_col else [
        "discussion_text", "lyric_text", "text", "content", "transcript"
    ]
    df, col = load_text_table(path, [c for c in candidates if c])
    texts = df[col].fillna("").astype(str).tolist()

    if args.fallback_tfidf:
        emb = encode_with_tfidf(texts)
        backend = "tfidf"
    else:
        try:
            emb = encode_with_st(texts, args.model)
            backend = args.model
        except Exception as e:
            print(f"sentence-transformers 失败 ({e})，回退 TF-IDF")
            emb = encode_with_tfidf(texts)
            backend = "tfidf-fallback"

    out_df = df.copy()
    out_df["__backend"] = backend
    out_df["__emb_dim"] = emb.shape[1]
    out_df["embedding"] = list(emb)

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(out_df)} rows, dim={emb.shape[1]}, backend={backend})")


if __name__ == "__main__":
    main()
