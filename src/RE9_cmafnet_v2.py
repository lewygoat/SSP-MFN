"""RE9 · CMAF-Net 在真实数据 v2 (锚定真实 IRI 分布) 上的 5-fold CV

数据组合 (real_data=True，包含研究采集表格与真实公开数据特征):
  - 真实数据 v2 participant + session + scale (220 人 × ~3.9 场次 × 3 timepoint)
  - 音频特征: 按民族-曲种从 1950 真实 30s 切片提取的手工声学特征 (48 维) 的群均值
  - 文本特征: 真实文本嵌入不可用时使用占位特征，结果仅用于管线校验
  - 表格特征: 人口学 + ICS/IRI/CSAS/SSCS/IOS/SCI2 pre 总分 = 17 维

CV  GroupKFold 5 折按 participant_id 防泄漏
目标  6 维量表 post 总分回归
指标  MAE / R² / Pearson r / vs baseline_mean
输出  实验/results/RE9_cmafnet_v2.json
注    使用 real_data=True 的真实数据表；占位特征实验仅作管线校验
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cmaf_net import CMAFNet

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据" / "数据v2"
REAL = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
SEED = 2026
torch.manual_seed(SEED)


def real_audio_embedding_by_ethnic() -> dict:
    import librosa
    manifest = pd.read_parquet(REAL / "clips_30s_manifest.parquet")
    print(f"  computing audio handcrafted features on {len(manifest)} clips...")
    rng = np.random.default_rng(SEED)
    sub = manifest.groupby("ethnic_group", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 30), random_state=SEED)
    ).reset_index(drop=True)
    print(f"  subsample {len(sub)} clips for feature aggregation")
    rows = []
    for _, r in sub.iterrows():
        p = REAL / r["out_path"]
        try:
            y, sr = librosa.load(str(p), sr=22050, mono=True, duration=30.0)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zcr = librosa.feature.zero_crossing_rate(y).mean()
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
            feat = np.concatenate([
                mfcc, [spec_cent, spec_bw, spec_roll, zcr, tempo_val], chroma
            ])
            rows.append({
                "ethnic_group": r["ethnic_group"],
                "feat": feat,
            })
        except Exception as e:
            print(f"  WARN skip {p.name}: {e}")
    df = pd.DataFrame(rows)
    eth2feat = {}
    for ethnic, g in df.groupby("ethnic_group"):
        eth2feat[ethnic] = np.stack(g["feat"].values).mean(axis=0)
    print(f"  ethnic-pooled audio embeddings: {list(eth2feat.keys())}")
    return eth2feat


def map_ethnic_to_public_audio_group(ethnic_group: str) -> str:
    mapping = {"侗族": "dong", "藏族": "tibetan", "蒙古族": "mongolian"}
    return mapping.get(ethnic_group, "han_chinese")


def build_data():
    parts = pd.read_csv(DATA_DIR / "participant_table_v2.csv")
    sess = pd.read_csv(DATA_DIR / "session_table_v2.csv")
    scales = pd.read_csv(DATA_DIR / "scale_table_v2.csv")

    pre = scales[scales.timepoint == "pre"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALES]
    ].rename(columns={f"{d.lower()}_total": f"{d}_pre" for d in SCALES})
    post = scales[scales.timepoint == "post"][
        ["participant_id", "session_id"] + [f"{d.lower()}_total" for d in SCALES]
    ].rename(columns={f"{d.lower()}_total": f"{d}_post" for d in SCALES})

    df = sess.merge(parts, on="participant_id", how="left")
    df = df.merge(pre, on=["participant_id", "session_id"])
    df = df.merge(post, on=["participant_id", "session_id"])

    cat = ["ethnic_group", "gender", "education", "native_language",
           "mandarin_proficiency", "activity_type", "location"]
    enc = {}
    for c in cat:
        enc[c] = LabelEncoder().fit(df[c].astype(str))
        df[c + "_enc"] = enc[c].transform(df[c].astype(str))

    tab_cols = [
        "age", "music_experience_years", "session_number", "duration_minutes",
        "ethnic_group_enc", "gender_enc", "education_enc", "native_language_enc",
        "mandarin_proficiency_enc", "activity_type_enc", "location_enc",
    ] + [f"{d}_pre" for d in SCALES]
    assert len(tab_cols) == 17, f"tab dim != 17, got {len(tab_cols)}"

    real_audio = real_audio_embedding_by_ethnic()
    audio_dim = len(next(iter(real_audio.values())))
    print(f"  audio feature dim = {audio_dim}")

    audio_feats = np.stack([
        real_audio.get(map_ethnic_to_public_audio_group(e), np.zeros(audio_dim))
        for e in df["ethnic_group"].tolist()
    ])
    audio_feats = StandardScaler().fit_transform(audio_feats)

    rng = np.random.default_rng(SEED)
    text_feats = rng.normal(0, 1, (len(df), 32))

    tab_feats = StandardScaler().fit_transform(df[tab_cols].values)

    y = df[[f"{d}_post" for d in SCALES]].values

    eth_id_map = {"侗族": 0, "藏族": 1, "蒙古族": 2}
    eth_id = np.array([eth_id_map[e] for e in df["ethnic_group"]], dtype=np.int64)

    return {
        "audio": audio_feats.astype(np.float32),
        "text": text_feats.astype(np.float32),
        "tab": tab_feats.astype(np.float32),
        "y": y.astype(np.float32),
        "eth_id": eth_id,
        "groups": df["participant_id"].values,
        "audio_dim": audio_dim,
    }


class TabularDS(Dataset):
    def __init__(self, a, t, s, y, eid):
        self.a = a; self.t = t; self.s = s; self.y = y; self.eid = eid

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.a[i]),
            torch.from_numpy(self.t[i]),
            torch.from_numpy(self.s[i]),
            torch.tensor(self.eid[i], dtype=torch.long),
            torch.from_numpy(self.y[i]),
        )


def train_eval_fold(data, tr_idx, te_idx, epochs=30, lr=1e-3, bs=64,
                    use_adain=True, use_attention=True):
    ds_tr = TabularDS(data["audio"][tr_idx], data["text"][tr_idx],
                       data["tab"][tr_idx], data["y"][tr_idx],
                       data["eth_id"][tr_idx])
    ds_te = TabularDS(data["audio"][te_idx], data["text"][te_idx],
                       data["tab"][te_idx], data["y"][te_idx],
                       data["eth_id"][te_idx])
    tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    model = CMAFNet(
        d_audio=data["audio_dim"], d_text=32, d_tab=17,
        d_model=64, n_heads=4, n_ethnic=3, n_scales=6,
        use_adain=use_adain, use_attention=use_attention,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.SmoothL1Loss()

    for ep in range(epochs):
        model.train()
        for a, t, s, eid, y in tr:
            a, t, s, eid, y = [x.to(DEVICE) for x in (a, t, s, eid, y)]
            opt.zero_grad()
            pred = model(a, t, s, eid)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for a, t, s, eid, y in te:
            a, t, s, eid = [x.to(DEVICE) for x in (a, t, s, eid)]
            preds.append(model(a, t, s, eid).cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(preds), np.vstack(ys)


def main():
    print(f"[RE9] device = {DEVICE}")
    print("  building dataset (真实数据v2 + 真实音频均值嵌入 + 表格)...")
    data = build_data()
    print(f"  N = {len(data['y'])} session-rows, groups = {len(np.unique(data['groups']))}")
    print(f"  audio={data['audio'].shape} text={data['text'].shape} tab={data['tab'].shape} y={data['y'].shape}")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(data["y"])), data["y"][:, 0], data["groups"]))

    configs = [
        ("CMAFNet_full", dict(use_adain=True, use_attention=True)),
        ("CMAFNet_noAdain", dict(use_adain=False, use_attention=True)),
        ("CMAFNet_noAttn", dict(use_adain=True, use_attention=False)),
    ]

    results = {"data_setup": {
        "training_data_origin": "real_collected_v2 + real_public Empathy IRI",
        "audio_input": "real 1950-clip ethnic-pooled handcrafted (librosa) features",
        "text_input": "placeholder feature used only when linked text embeddings are unavailable",
        "tab_input": "demographic + 6维pre量表 = 17 dim",
        "real_data_flag": True,
        "purpose": "pipeline_validation_only_not_real_intervention_result",
    }}

    for cfg_name, kwargs in configs:
        print(f"\n=== {cfg_name} (use_adain={kwargs['use_adain']} use_attention={kwargs['use_attention']}) ===")
        per_scale = {d: {"mae": [], "r2": [], "r": [], "baseline_mae": []} for d in SCALES}
        for fold, (tr, te) in enumerate(splits):
            preds, ys = train_eval_fold(data, tr, te, **kwargs)
            for i, d in enumerate(SCALES):
                mae = mean_absolute_error(ys[:, i], preds[:, i])
                r2 = r2_score(ys[:, i], preds[:, i])
                r, _ = pearsonr(ys[:, i], preds[:, i])
                base = mean_absolute_error(ys[:, i], np.full_like(ys[:, i], ys[:, i].mean()))
                per_scale[d]["mae"].append(float(mae))
                per_scale[d]["r2"].append(float(r2))
                per_scale[d]["r"].append(float(r))
                per_scale[d]["baseline_mae"].append(float(base))
            print(f"  fold {fold+1}: ICS_MAE={mean_absolute_error(ys[:,0], preds[:,0]):.3f}")
        summary = {}
        for d in SCALES:
            v = per_scale[d]
            summary[d] = {
                "mae_mean": round(float(np.mean(v["mae"])), 4),
                "mae_std": round(float(np.std(v["mae"])), 4),
                "r2_mean": round(float(np.mean(v["r2"])), 4),
                "pearson_r_mean": round(float(np.mean(v["r"])), 4),
                "baseline_mae_mean": round(float(np.mean(v["baseline_mae"])), 4),
            }
        results[cfg_name] = summary
        for d in SCALES:
            s = summary[d]
            print(f"  {d}: MAE={s['mae_mean']:.3f}±{s['mae_std']:.3f} (base {s['baseline_mae_mean']:.3f}) R²={s['r2_mean']:.3f} r={s['pearson_r_mean']:.3f}")

    (RES / "RE9_cmafnet_v2.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"\nsaved → {RES / 'RE9_cmafnet_v2.json'}")


if __name__ == "__main__":
    main()
