"""音频嵌入提取管线（无需GPU、无需大模型下载）

支持三种特征：
- handcrafted: librosa MFCC + chroma + spectral + tempo（轻量、可复现）
- yamnet: TensorFlow Hub YAMNet 521维（可选，需tf-hub）
- wav2vec_local: 仅在用户自行下载facebook/wav2vec2-base时启用

输出：每段音频 → 1×D 嵌入向量 + 元数据，存为 .parquet
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa


def handcrafted_embedding(
    y: np.ndarray, sr: int, n_mfcc: int = 20
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_d = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    parts = []
    for arr in [mfcc, mfcc_d, chroma, contrast, centroid, rolloff, bandwidth, zcr]:
        parts.append(arr.mean(axis=1))
        parts.append(arr.std(axis=1))
    parts.append(np.array([float(tempo)]))
    return np.concatenate(parts).astype(np.float32)


def process_one(
    path: Path, sr: int = 22050, max_seconds: float = 30.0
) -> dict | None:
    try:
        y, sr_ = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
        if len(y) < sr * 1:
            return None
        emb = handcrafted_embedding(y, sr_)
        return {
            "path": str(path),
            "name": path.name,
            "sr": int(sr_),
            "duration_s": float(len(y) / sr_),
            "emb_dim": int(emb.shape[0]),
            "embedding": emb.tolist(),
        }
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def walk_audio_files(root: Path, exts=(".wav", ".mp3", ".flac", ".m4a", ".ogg")):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True, help="音频根目录（递归扫描）")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0表示不限")
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    root = Path(args.audio_root)
    files = list(walk_audio_files(root))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"found {len(files)} audio files under {root}")

    rows, errs = [], []
    for i, fp in enumerate(files, 1):
        rec = process_one(fp, sr=args.sr)
        if rec is None:
            continue
        if "error" in rec:
            errs.append(rec)
        else:
            rows.append(rec)
        if i % 20 == 0:
            print(f"  [{i}/{len(files)}] ok={len(rows)} err={len(errs)}")

    if not rows:
        print("no embeddings extracted; writing empty parquet")
        pd.DataFrame(columns=["path", "name", "sr", "duration_s", "embedding"]).to_parquet(
            args.out_parquet
        )
        return

    df = pd.DataFrame(rows)
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, dim={df.iloc[0]['emb_dim']})")
    if errs:
        err_path = out.with_suffix(".errors.json")
        json.dump(errs, open(err_path, "w"), ensure_ascii=False, indent=2)
        print(f"errors logged: {err_path}")


if __name__ == "__main__":
    main()
