"""RE1 · MERT-v1-95M 在 1950 切片上的嵌入 + UMAP 可视化

输入  clips_30s_manifest.parquet (1950 切片 × 30s × 22050 Hz)
模型  m-a-p/MERT-v1-95M (HuggingFace, ~95M params)
特征  取所有隐藏层 mean-of-mean → (n_layers+1, 768) → 时间-层 双重池化 → 768 维
后处理  UMAP 2D + StandardScaler
输出
  features/RE1_mert_embeddings.npz   shape=(N, 768) + meta
  results/RE1_mert_umap.json         按民族/语料库的中心点统计
  features/RE1_mert_umap.parquet     UMAP 2D 坐标 + 元数据
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
OUT = ROOT / "数据" / "真实数据集成" / "output"
FEAT = ROOT / "实验" / "features"
RES = ROOT / "实验" / "results"
FEAT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main():
    from transformers import AutoModel, AutoFeatureExtractor
    print(f"[RE1] device = {DEVICE}")
    print("  loading MERT-v1-95M...")
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model.eval().to(DEVICE)
    fe = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    target_sr = int(fe.sampling_rate)
    print(f"  MERT target_sr = {target_sr}")

    manifest = pd.read_parquet(OUT / "clips_30s_manifest.parquet")
    print(f"  manifest n = {len(manifest)}")

    embeddings = np.zeros((len(manifest), 768), dtype=np.float32)
    failed = []
    resamplers = {}
    for i, r in enumerate(manifest.itertuples()):
        p = OUT / r.out_path
        try:
            wav, sr = torchaudio.load(str(p))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
                wav = resamplers[sr](wav)
            wav = wav.squeeze(0).numpy().astype(np.float32)
            wav = wav / (np.abs(wav).max() + 1e-8)
            inputs = fe(wav, sampling_rate=target_sr, return_tensors="pt")
            with torch.no_grad():
                out = model(**{k: v.to(DEVICE) for k, v in inputs.items()},
                            output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)
            emb = hs.mean(dim=2).mean(dim=0).squeeze(0).cpu().numpy()
            embeddings[i] = emb
        except Exception as e:
            failed.append((p.name, str(e)))
            embeddings[i] = np.nan
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(manifest)}")

    mask = ~np.isnan(embeddings).any(axis=1)
    valid = manifest[mask].reset_index(drop=True).copy()
    emb_valid = embeddings[mask]
    print(f"  valid embeddings: {len(valid)} / {len(manifest)} (failed {len(failed)})")

    np.savez_compressed(
        FEAT / "RE1_mert_embeddings.npz",
        embeddings=emb_valid.astype(np.float32),
        clip_uid=valid["clip_uid"].to_numpy(),
        corpus=valid["corpus"].to_numpy(),
        ethnic_group=valid["ethnic_group"].to_numpy(),
        genre=valid["genre"].to_numpy(),
    )

    print("  running UMAP 2D...")
    from sklearn.preprocessing import StandardScaler
    import umap
    scaled = StandardScaler().fit_transform(emb_valid)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=2026)
    coords = reducer.fit_transform(scaled)
    valid["umap_x"] = coords[:, 0]
    valid["umap_y"] = coords[:, 1]
    valid.to_parquet(FEAT / "RE1_mert_umap.parquet", index=False)

    centers = valid.groupby("ethnic_group").agg(
        n=("clip_uid", "count"),
        umap_x_mean=("umap_x", "mean"),
        umap_y_mean=("umap_y", "mean"),
        umap_x_std=("umap_x", "std"),
        umap_y_std=("umap_y", "std"),
    ).round(4).reset_index()
    cent_corpus = valid.groupby("corpus").agg(
        n=("clip_uid", "count"),
        umap_x_mean=("umap_x", "mean"),
        umap_y_mean=("umap_y", "mean"),
    ).round(4).reset_index()

    from sklearn.metrics import silhouette_score
    eth_labels = pd.Categorical(valid["ethnic_group"]).codes
    corpus_labels = pd.Categorical(valid["corpus"]).codes
    sil_ethnic = float(silhouette_score(emb_valid, eth_labels, metric="cosine"))
    sil_corpus = float(silhouette_score(emb_valid, corpus_labels, metric="cosine"))

    summary = {
        "data_source": "1950 real 30s clips from {Bai, CCMUSIC, Choral, Dong/Tibetan YT, Mongolian YT}",
        "model": "m-a-p/MERT-v1-95M, mean over hidden states then time",
        "n_valid_embeddings": int(len(valid)),
        "embedding_dim": int(emb_valid.shape[1]),
        "umap_silhouette_ethnic": round(sil_ethnic, 4),
        "umap_silhouette_corpus": round(sil_corpus, 4),
        "ethnic_centers": centers.to_dict(orient="records"),
        "corpus_centers": cent_corpus.to_dict(orient="records"),
        "n_failed": len(failed),
    }
    (RES / "RE1_mert_umap.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print(f"\n  按民族 UMAP 中心:")
    for r in summary["ethnic_centers"]:
        print(f"    {r['ethnic_group']:<28} n={r['n']:>4}  μ=({r['umap_x_mean']:+.2f}, {r['umap_y_mean']:+.2f})")
    print(f"  silhouette(ethnic) = {sil_ethnic:.4f}")
    print(f"  silhouette(corpus) = {sil_corpus:.4f}")
    print(f"saved → {RES / 'RE1_mert_umap.json'}, {FEAT / 'RE1_mert_embeddings.npz'}")


if __name__ == "__main__":
    main()
