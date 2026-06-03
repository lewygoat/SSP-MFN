"""RE7 · Choral Singing Dataset 4声部协同分析

数据  ChoralSingingDataset (Zenodo 2649950, CC-BY-4.0)
      3 曲目 (Locus Iste / El Rossinyol / Nino Dios) × 4 声部 (S/A/T/B) × 4 take = 48 文件
任务  每个文件提取 F0/RMS/spectral_centroid, 计算同 piece+take 内 4 声部 F0 相关矩阵 (合唱协同)
输出  实验/results/RE7_choral_sync.json
      实验/features/RE7_choral_per_file.parquet
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from itertools import combinations
from scipy import stats

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
CSD = ROOT / "数据" / "真实锚定数据集" / "audio_corpora" / "choral_singing" / "ChoralSingingDataset"
RES = ROOT / "实验" / "results"
FEAT = ROOT / "实验" / "features"
RES.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)

VOICES = ["soprano", "alto", "tenor", "bass"]


def parse_name(p: Path) -> dict:
    m = re.match(r"CSD_(\w+)_(\w+)_(\d+)\.wav$", p.name)
    if not m:
        return {}
    return dict(piece=m.group(1), voice=m.group(2), take=int(m.group(3)))


def extract_acoustic(p: Path, sr: int = 22050) -> dict:
    y, sr = librosa.load(str(p), sr=sr, mono=True)
    f0_series, voiced, prob = librosa.pyin(
        y, fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr, frame_length=2048, hop_length=512,
    )
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]

    f0_valid = f0_series[~np.isnan(f0_series)]
    return {
        "duration_sec": float(round(len(y) / sr, 3)),
        "n_voiced_frames": int(np.sum(voiced)),
        "voiced_ratio": float(round(np.mean(voiced), 4)),
        "f0_mean_hz": float(round(f0_valid.mean(), 2)) if len(f0_valid) else np.nan,
        "f0_median_hz": float(round(np.median(f0_valid), 2)) if len(f0_valid) else np.nan,
        "f0_std_hz": float(round(f0_valid.std(), 2)) if len(f0_valid) else np.nan,
        "f0_min_hz": float(round(f0_valid.min(), 2)) if len(f0_valid) else np.nan,
        "f0_max_hz": float(round(f0_valid.max(), 2)) if len(f0_valid) else np.nan,
        "rms_mean": float(round(rms.mean(), 4)),
        "rms_std": float(round(rms.std(), 4)),
        "spec_centroid_mean_hz": float(round(spec_cent.mean(), 2)),
        "spec_centroid_std_hz": float(round(spec_cent.std(), 2)),
        "_f0_track": f0_series,
        "_voiced_track": voiced.astype(np.float32),
    }


def voice_pair_correlation(f0_dict: dict) -> dict:
    pairs = {}
    voices_present = list(f0_dict.keys())
    for v1, v2 in combinations(voices_present, 2):
        a = f0_dict[v1]
        b = f0_dict[v2]
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 30:
            pairs[f"{v1}_vs_{v2}"] = {"n_voiced_overlap": int(mask.sum()), "r": np.nan, "p": np.nan}
            continue
        log_a = np.log2(a[mask])
        log_b = np.log2(b[mask])
        r, p = stats.pearsonr(log_a, log_b)
        pairs[f"{v1}_vs_{v2}"] = {
            "n_voiced_overlap": int(mask.sum()),
            "r": float(round(r, 4)),
            "p": float(p),
        }
    return pairs


def main():
    files = sorted([p for p in CSD.glob("CSD_*.wav") if not p.name.startswith("._")])
    print(f"[RE7] choral files = {len(files)}")
    rows = []
    f0_store = {}
    for i, p in enumerate(files):
        meta = parse_name(p)
        if not meta:
            continue
        feat = extract_acoustic(p)
        rows.append({**meta, "file": p.name, **{k: v for k, v in feat.items() if not k.startswith("_")}})
        key = (meta["piece"], meta["take"])
        if key not in f0_store:
            f0_store[key] = {}
        f0_store[key][meta["voice"]] = feat["_f0_track"]
        if (i + 1) % 8 == 0:
            print(f"  processed {i+1}/{len(files)}: piece={meta['piece']} voice={meta['voice']} take={meta['take']}")

    df = pd.DataFrame(rows)
    df.to_parquet(FEAT / "RE7_choral_per_file.parquet", index=False)

    voice_stats = df.groupby("voice").agg(
        n=("file", "count"),
        f0_mean=("f0_mean_hz", "mean"),
        f0_std=("f0_mean_hz", "std"),
    ).round(2).to_dict(orient="index")

    sync_results = {}
    for (piece, take), voice_f0 in f0_store.items():
        sync_results[f"{piece}_take{take}"] = voice_pair_correlation(voice_f0)

    pair_rs = []
    for k, pairs in sync_results.items():
        for pname, p in pairs.items():
            if np.isfinite(p.get("r", np.nan)):
                pair_rs.append({"piece_take": k, "pair": pname, "r": p["r"]})
    pair_r_df = pd.DataFrame(pair_rs)
    pair_summary = pair_r_df.groupby("pair")["r"].agg(["mean", "std", "min", "max", "count"]).round(4)

    result = {
        "data_source": "Zenodo 2649950 ChoralSingingDataset, CC-BY-4.0",
        "n_files_processed": int(len(df)),
        "voice_acoustic_stats": voice_stats,
        "per_piece_take_voice_correlation": sync_results,
        "pair_r_summary": pair_summary.reset_index().to_dict(orient="records"),
        "interpretation": {
            "high_r": "同 piece+take 内某两个声部 log-F0 高相关 → 协同/同步性强",
            "expected_pairs": "soprano-alto / tenor-bass 一般 r 较高 (同性别声区)",
        },
    }
    (RES / "RE7_choral_sync.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: float(x) if np.isfinite(x) else None)
    )

    print(f"\n  按声部声学统计:")
    for v, s in voice_stats.items():
        print(f"    {v:<10} n={s['n']}  f0_mean={s['f0_mean']:.1f}±{s['f0_std']:.1f} Hz")
    print(f"\n  声部对 log-F0 相关 (mean over 12 piece-take blocks):")
    for r in result["pair_r_summary"]:
        print(f"    {r['pair']:<24} r={r['mean']:.3f}±{r['std']:.3f}  (n_block={int(r['count'])})")
    print(f"saved → {RES / 'RE7_choral_sync.json'}")


if __name__ == "__main__":
    main()
