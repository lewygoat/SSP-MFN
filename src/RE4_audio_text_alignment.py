"""RE4 · 跨模态对齐: Bai Dabenqu 550 句 MP3 ↔ 1574 phrase 歌词

任务  给定一段音频, 从 N 个候选歌词中检索对应文本; vice versa
模型  MERT-v1-95M (audio) + XLM-RoBERTa-base (text), 双塔零样本对齐
指标  recall@K (K=1, 5, 10), median rank
输出  实验/results/RE4_audio_text_alignment.json
      实验/features/RE4_bai_paired_emb.npz
"""
from __future__ import annotations
import json
import os
import re
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
BAI_AUDIO = (ROOT / "数据" / "真实锚定数据集" / "audio_corpora" /
             "bai_dabenqu" / "bai_audio_extracted")
OUT_DATA = ROOT / "数据" / "真实数据集成" / "output"
FEAT = ROOT / "实验" / "features"
RES = ROOT / "实验" / "results"
FEAT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def find_audio_dir() -> Path:
    for sub in BAI_AUDIO.iterdir():
        if sub.is_dir() and not sub.name.startswith("._"):
            return sub
    return BAI_AUDIO


def load_audio_files() -> pd.DataFrame:
    d = find_audio_dir()
    rows = []
    for p in sorted(d.iterdir()):
        if not p.suffix.lower() == ".mp3":
            continue
        if p.name.startswith("._"):
            continue
        m = re.match(r"^\s*(\d+)\s*\.mp3$", p.name)
        if not m:
            continue
        rows.append({"seq_num": int(m.group(1)), "path": p})
    return pd.DataFrame(rows).sort_values("seq_num").reset_index(drop=True)


def encode_mert(paths, model, fe, target_sr=24000, max_sec=10.0):
    embs = []
    resamp = {}
    for p in paths:
        wav, sr = torchaudio.load(str(p))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            if sr not in resamp:
                resamp[sr] = torchaudio.transforms.Resample(sr, target_sr)
            wav = resamp[sr](wav)
        wav = wav.squeeze(0).numpy().astype(np.float32)
        if len(wav) > int(max_sec * target_sr):
            wav = wav[:int(max_sec * target_sr)]
        wav = wav / (np.abs(wav).max() + 1e-8)
        inputs = fe(wav, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.to(DEVICE) for k, v in inputs.items()},
                        output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)
        emb = hs.mean(dim=2).mean(dim=0).squeeze(0).cpu().numpy()
        embs.append(emb)
    return np.array(embs)


def encode_text(texts, model, tok, bs=32, max_len=64):
    model.eval()
    embs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)


def retrieval_metrics(sim: np.ndarray) -> dict:
    n = sim.shape[0]
    ranks_a2t, ranks_t2a = [], []
    for i in range(n):
        order = np.argsort(-sim[i])
        r = int(np.where(order == i)[0][0])
        ranks_a2t.append(r)
    sim_T = sim.T
    for i in range(n):
        order = np.argsort(-sim_T[i])
        r = int(np.where(order == i)[0][0])
        ranks_t2a.append(r)
    r_a2t = np.array(ranks_a2t)
    r_t2a = np.array(ranks_t2a)
    def stats(r):
        return {
            "recall@1": float(round((r < 1).mean(), 4)),
            "recall@5": float(round((r < 5).mean(), 4)),
            "recall@10": float(round((r < 10).mean(), 4)),
            "median_rank": int(np.median(r)),
            "mean_rank": float(round(r.mean(), 2)),
        }
    return {"audio2text": stats(r_a2t), "text2audio": stats(r_t2a), "n": n}


def main():
    from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
    print(f"[RE4] device = {DEVICE}")

    audio_df = load_audio_files()
    print(f"  found {len(audio_df)} Bai MP3 files")

    phrase = pd.read_parquet(OUT_DATA / "real_lyric_table_phrase.parquet")
    phrase = phrase.copy()
    phrase = phrase.reset_index(drop=True)
    phrase["seq_num"] = phrase.index + 1
    paired = audio_df.merge(phrase, on="seq_num", how="inner")
    print(f"  paired = {len(paired)} audio-text pairs")

    np.random.seed(2026)
    if len(paired) > 200:
        keep_idx = np.random.choice(len(paired), 200, replace=False)
        paired = paired.iloc[sorted(keep_idx)].reset_index(drop=True)
    print(f"  using N = {len(paired)} for retrieval evaluation")

    print("  loading MERT-v1-95M...")
    audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M",
                                              trust_remote_code=True).to(DEVICE).eval()
    audio_fe = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",
                                                      trust_remote_code=True)

    print(f"  encoding {len(paired)} audio with MERT...")
    emb_a = encode_mert(paired["path"].tolist(), audio_model, audio_fe)
    print(f"  emb_a shape = {emb_a.shape}")

    print("  loading XLM-RoBERTa-base...")
    text_model = AutoModel.from_pretrained("xlm-roberta-base").to(DEVICE).eval()
    text_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    print("  encoding text with XLM-R (Chinese)...")
    emb_t_zh = encode_text(paired["chinese_text"].astype(str).tolist(),
                            text_model, text_tok)
    print("  encoding text with XLM-R (Bai script)...")
    emb_t_eth = encode_text(paired["ethnic_script"].astype(str).tolist(),
                             text_model, text_tok)
    print("  encoding text with XLM-R (English)...")
    emb_t_en = encode_text(paired["english_text"].fillna("").astype(str).tolist(),
                            text_model, text_tok)

    np.savez_compressed(
        FEAT / "RE4_bai_paired_emb.npz",
        emb_audio=emb_a.astype(np.float32),
        emb_text_zh=emb_t_zh.astype(np.float32),
        emb_text_eth=emb_t_eth.astype(np.float32),
        emb_text_en=emb_t_en.astype(np.float32),
        seq_num=paired["seq_num"].to_numpy(),
        song_title=paired["song_title"].to_numpy(),
    )

    def cosine_sim(A, B):
        A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_n @ B_n.T

    results = {
        "data_source": "Bai Dabenqu Zenodo 16746020 (CC-BY-4.0)",
        "n_pairs_used": int(len(paired)),
        "audio_model": "m-a-p/MERT-v1-95M",
        "text_model": "xlm-roberta-base",
        "task": "zero-shot audio-text retrieval (no contrastive training, frozen encoders)",
        "by_text_language": {},
    }

    for lang, emb_t in [("chinese", emb_t_zh), ("bai_script", emb_t_eth),
                         ("english", emb_t_en)]:
        sim = cosine_sim(emb_a, emb_t)
        m = retrieval_metrics(sim)
        results["by_text_language"][lang] = m
        print(f"\n  {lang}:")
        print(f"    audio→text  R@1={m['audio2text']['recall@1']:.4f}  R@5={m['audio2text']['recall@5']:.4f}  R@10={m['audio2text']['recall@10']:.4f}  median={m['audio2text']['median_rank']}")
        print(f"    text→audio  R@1={m['text2audio']['recall@1']:.4f}  R@5={m['text2audio']['recall@5']:.4f}  R@10={m['text2audio']['recall@10']:.4f}  median={m['text2audio']['median_rank']}")

    chance_r1 = round(1.0 / len(paired), 4)
    results["chance_recall@1"] = chance_r1
    print(f"\n  chance recall@1 = {chance_r1}")

    (RES / "RE4_audio_text_alignment.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"saved → {RES / 'RE4_audio_text_alignment.json'}")


if __name__ == "__main__":
    main()
