"""RE3 · XLM-RoBERTa 在 Bai Dabenqu 多语言歌词上的 phrase 级 song 分类

输入  real_lyric_table_phrase.parquet (1574 条 phrase, 5 剧目)
模型  xlm-roberta-base, [CLS] pooling, frozen + linear probe
任务  剧目(song_title)分类 (主测试: 跨语言文本是否携带剧目身份信号)
CV    StratifiedKFold 5 折
输出  实验/results/RE3_xlmr_lyric_cls.json
      实验/features/RE3_xlmr_text_emb.npz
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
OUT = ROOT / "数据" / "真实数据集成" / "output"
FEAT = ROOT / "实验" / "features"
RES = ROOT / "实验" / "results"
FEAT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def encode_xlmr(texts, model, tok, batch_size=32, max_len=64):
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(cls)
    return np.vstack(embs)


def main():
    from transformers import AutoModel, AutoTokenizer
    print(f"[RE3] device = {DEVICE}")

    phrase = pd.read_parquet(OUT / "real_lyric_table_phrase.parquet")
    phrase = phrase[phrase["song_title"] != "其他"].reset_index(drop=True)
    cnt = phrase["song_title"].value_counts()
    keep = cnt[cnt >= 50].index.tolist()
    phrase = phrase[phrase["song_title"].isin(keep)].reset_index(drop=True)
    print(f"  n_phrase = {len(phrase)}, songs kept (>=50) = {len(keep)}")
    print(f"  songs: {list(keep)}")

    model = AutoModel.from_pretrained("xlm-roberta-base").to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print("  XLM-R loaded.")

    print("  encoding 中文 (chinese_text)...")
    zh_texts = phrase["chinese_text"].astype(str).tolist()
    emb_zh = encode_xlmr(zh_texts, model, tok)
    print(f"  emb_zh shape = {emb_zh.shape}")

    print("  encoding 民族语言原文 (ethnic_script) where present...")
    eth_mask = phrase["script_has_native"].values
    eth_texts_arr = phrase["ethnic_script"].astype(str).tolist()
    emb_eth = encode_xlmr(eth_texts_arr, model, tok)
    print(f"  emb_eth shape = {emb_eth.shape}")

    print("  encoding 英文 (english_text)...")
    en_texts = phrase["english_text"].fillna("").astype(str).tolist()
    emb_en = encode_xlmr(en_texts, model, tok)
    print(f"  emb_en shape = {emb_en.shape}")

    np.savez_compressed(
        FEAT / "RE3_xlmr_text_emb.npz",
        emb_zh=emb_zh.astype(np.float32),
        emb_eth=emb_eth.astype(np.float32),
        emb_en=emb_en.astype(np.float32),
        lyric_uid=phrase["lyric_uid"].to_numpy(),
        song_title=phrase["song_title"].to_numpy(),
        script_has_native=eth_mask.astype(np.int8),
    )

    le = LabelEncoder().fit(phrase["song_title"])
    y = le.transform(phrase["song_title"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

    results = {
        "data_source": "Zenodo 16746020 Bai Dabenqu, CC-BY-4.0",
        "n_phrase": int(len(phrase)),
        "n_songs": int(len(le.classes_)),
        "song_classes": le.classes_.tolist(),
        "language_evaluation": {},
    }

    for lang, X in [("chinese", emb_zh), ("ethnic_bai", emb_eth), ("english", emb_en)]:
        if lang == "ethnic_bai":
            mask = eth_mask
            X_use = X[mask]
            y_use = y[mask]
        else:
            X_use = X
            y_use = y
        acc_list, f1_list = [], []
        for tr, te in skf.split(X_use, y_use):
            sc = StandardScaler().fit(X_use[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced", n_jobs=1)
            clf.fit(sc.transform(X_use[tr]), y_use[tr])
            preds = clf.predict(sc.transform(X_use[te]))
            acc_list.append(accuracy_score(y_use[te], preds))
            f1_list.append(f1_score(y_use[te], preds, average="macro", zero_division=0))
        results["language_evaluation"][lang] = {
            "n_samples": int(len(y_use)),
            "acc_mean": round(float(np.mean(acc_list)), 4),
            "acc_std": round(float(np.std(acc_list)), 4),
            "f1_macro_mean": round(float(np.mean(f1_list)), 4),
            "f1_macro_std": round(float(np.std(f1_list)), 4),
            "chance_acc": round(1.0 / len(le.classes_), 4),
        }
        print(f"  {lang:<12} acc={results['language_evaluation'][lang]['acc_mean']:.4f}±{results['language_evaluation'][lang]['acc_std']:.4f}  f1={results['language_evaluation'][lang]['f1_macro_mean']:.4f}  n={len(y_use)}")

    (RES / "RE3_xlmr_lyric_cls.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"saved → {RES / 'RE3_xlmr_lyric_cls.json'}")


if __name__ == "__main__":
    main()
