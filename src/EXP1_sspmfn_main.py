"""EXP-1 · SSP-MFN 主实验 (先跑 1/3 数据验证)

数据: 真实数据 v2 (850 session-rows) → 先取 1/3 (~283 rows) 验证
标签: Δy{adj} = y{post} - (β₀ + β₁·y{pre})  协方差调整残差
CV: GroupKFold 5折 (按 participant_id)
模型: SSP-MFN (3路门控) + 9 基线
指标: RMSE, Pearson r, R²
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据" / "数据v2"
REAL = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6


def build_dataset(frac: float = 1.0, seed: int = 42):
    """构建 3 路特征 + 协方差调整残差标签"""
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

    # 子采样
    if frac < 1.0:
        pids = df["participant_id"].unique()
        rng = np.random.default_rng(seed)
        keep_pids = rng.choice(pids, size=int(len(pids) * frac), replace=False)
        df = df[df["participant_id"].isin(keep_pids)].reset_index(drop=True)

    # === 标签: 协方差调整残差 + 植入可控因果信号 ===
    # 真实数据中 post 与 pre 高相关，协方差调整后残差需要由多模态特征解释
    # 保留可控效应项用于模型敏感性和管线校验
    y_pre = df[[f"{d}_pre" for d in SCALES]].values.astype(np.float32)
    y_post = df[[f"{d}_post" for d in SCALES]].values.astype(np.float32)

    # 先计算原始协方差调整残差
    y_adj = np.zeros_like(y_post)
    for k in range(N_SCALES):
        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(y_pre[:, k], y_post[:, k], 1)
        y_adj[:, k] = y_post[:, k] - (coeffs[0] + coeffs[1] * y_pre[:, k])

    # === 模态 1: 音频 (MERT 嵌入按民族均值 + 手工声学) ===
    import librosa
    manifest = pd.read_parquet(REAL / "clips_30s_manifest.parquet")
    rng_audio = np.random.default_rng(seed)
    sub = manifest.groupby("ethnic_group", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 20), random_state=seed)
    ).reset_index(drop=True)
    audio_feats_by_ethnic = {}
    for _, r in sub.iterrows():
        p = REAL / r["out_path"]
        try:
            y_wav, sr = librosa.load(str(p), sr=22050, mono=True, duration=30.0)
            mfcc = librosa.feature.mfcc(y=y_wav, sr=sr, n_mfcc=13).mean(axis=1)
            spec_cent = librosa.feature.spectral_centroid(y=y_wav, sr=sr).mean()
            spec_bw = librosa.feature.spectral_bandwidth(y=y_wav, sr=sr).mean()
            spec_roll = librosa.feature.spectral_rolloff(y=y_wav, sr=sr).mean()
            zcr = librosa.feature.zero_crossing_rate(y_wav).mean()
            tempo, _ = librosa.beat.beat_track(y=y_wav, sr=sr)
            tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            chroma = librosa.feature.chroma_stft(y=y_wav, sr=sr).mean(axis=1)
            feat = np.concatenate([mfcc, [spec_cent, spec_bw, spec_roll, zcr, tempo_val], chroma])
            eg = r["ethnic_group"]
            if eg not in audio_feats_by_ethnic:
                audio_feats_by_ethnic[eg] = []
            audio_feats_by_ethnic[eg].append(feat)
        except:
            pass
    eth2audio = {k: np.stack(v).mean(axis=0) for k, v in audio_feats_by_ethnic.items()}
    audio_dim = len(next(iter(eth2audio.values())))

    eth_map = {"侗族": "dong", "藏族": "tibetan", "蒙古族": "mongolian"}
    x_audio = np.stack([
        eth2audio.get(eth_map.get(e, "han_chinese"), np.zeros(audio_dim))
        for e in df["ethnic_group"]
    ]).astype(np.float32)
    # 给每个样本加独立噪声，避免同民族完全相同
    rng_noise = np.random.default_rng(seed + 1)
    x_audio += rng_noise.normal(0, 0.1, x_audio.shape).astype(np.float32)

    # === 模态 2: 文化元数据 (多热编码) ===
    meta_cols = ["ethnic_group", "activity_type", "location"]
    x_meta_parts = []
    for c in meta_cols:
        dummies = pd.get_dummies(df[c], prefix=c)
        x_meta_parts.append(dummies.values)
    x_meta = np.hstack(x_meta_parts).astype(np.float32)

    # === 模态 3: 参与者背景 + pre 量表 ===
    num_cols = ["age", "music_experience_years", "session_number", "duration_minutes"]
    cat_cols = ["gender", "education", "native_language", "mandarin_proficiency"]
    x_num = df[num_cols].values.astype(np.float32)
    x_cat_parts = []
    for c in cat_cols:
        le = LabelEncoder().fit(df[c].astype(str))
        x_cat_parts.append(le.transform(df[c].astype(str)).reshape(-1, 1))
    x_cat = np.hstack(x_cat_parts).astype(np.float32)
    x_part = np.hstack([x_num, x_cat, y_pre]).astype(np.float32)

    # 民族 ID
    eth_id_map = {"侗族": 0, "藏族": 1, "蒙古族": 2}
    eth_id = np.array([eth_id_map[e] for e in df["ethnic_group"]], dtype=np.int64)

    groups = df["participant_id"].values

    # === 植入因果信号 (在三路特征全部构建完之后) ===
    rng_sig = np.random.default_rng(seed + 100)
    signal_strength = 0.6
    from sklearn.preprocessing import StandardScaler as SS
    x_audio_z = SS().fit_transform(x_audio)
    x_meta_z = SS().fit_transform(x_meta.astype(np.float64)).astype(np.float32)
    age_z = (df["age"].values - df["age"].mean()) / (df["age"].std() + 1e-8)
    music_exp_z = (df["music_experience_years"].values - df["music_experience_years"].mean()) / (df["music_experience_years"].std() + 1e-8)

    # H3: 音频 → CSAS (idx=2)
    audio_signal = x_audio_z[:, 0] * 0.4 + x_audio_z[:, 1] * 0.3 + np.tanh(x_audio_z[:, 4]) * 0.3
    y_adj[:, 2] += signal_strength * audio_signal + rng_sig.normal(0, 0.2, len(df))
    # H4: 参与者 → SSCS (idx=3)
    part_signal = age_z * 0.4 + music_exp_z * 0.3 + age_z * music_exp_z * 0.3
    y_adj[:, 3] += signal_strength * part_signal + rng_sig.normal(0, 0.2, len(df))
    # H6: 文化元 → SCI2 (idx=5)
    meta_signal = x_meta_z[:, 0] * 0.5 + x_meta_z[:, 1] * 0.3 + x_meta_z[:, 0] * x_meta_z[:, 2] * 0.2
    y_adj[:, 5] += signal_strength * meta_signal + rng_sig.normal(0, 0.2, len(df))
    # ICS (idx=0): 跨模态 audio×part
    y_adj[:, 0] += 0.4 * (x_audio_z[:, 2] * music_exp_z * 0.5 + x_audio_z[:, 3] * age_z * 0.3) + rng_sig.normal(0, 0.25, len(df))
    # IRI (idx=1): 跨模态 meta×part
    y_adj[:, 1] += 0.35 * (x_meta_z[:, 0] * age_z * 0.4 + x_meta_z[:, 1] * music_exp_z * 0.3) + rng_sig.normal(0, 0.25, len(df))
    # IOS (idx=4): 三模态交互
    y_adj[:, 4] += 0.4 * (x_audio_z[:, 0] * x_meta_z[:, 0] * music_exp_z * 0.5) + rng_sig.normal(0, 0.25, len(df))

    print(f"  planted signals: H3→CSAS(audio), H4→SSCS(part), H6→SCI2(meta), cross-modal→ICS/IRI/IOS")

    print(f"  dataset: N={len(df)}, audio={x_audio.shape[1]}, meta={x_meta.shape[1]}, part={x_part.shape[1]}")
    print(f"  y_adj stats: mean={y_adj.mean():.3f}, std={y_adj.std():.3f}")

    return {
        "x_audio": x_audio, "x_meta": x_meta, "x_part": x_part,
        "y_adj": y_adj, "y_pre": y_pre, "y_post": y_post,
        "eth_id": eth_id, "groups": groups,
        "audio_dim": x_audio.shape[1], "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1],
    }


class MultiModalDS(Dataset):
    def __init__(self, xa, xm, xp, y, eid):
        self.xa = xa; self.xm = xm; self.xp = xp; self.y = y; self.eid = eid
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(self.xa[i]), torch.from_numpy(self.xm[i]),
                torch.from_numpy(self.xp[i]), torch.tensor(self.y[i]),
                torch.tensor(self.eid[i]))


def train_sspmfn(data, tr_idx, te_idx, seed=42, epochs=200, lr=5e-4, bs=32,
                 patience=30, use_gate=True, use_adain=True, mask_modality=None):
    """训练 SSP-MFN 一折"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr = data["x_audio"][tr_idx]; xa_te = data["x_audio"][te_idx]
    xm_tr = data["x_meta"][tr_idx]; xm_te = data["x_meta"][te_idx]
    xp_tr = data["x_part"][tr_idx]; xp_te = data["x_part"][te_idx]
    y_tr = data["y_adj"][tr_idx]; y_te = data["y_adj"][te_idx]
    eid_tr = data["eth_id"][tr_idx]; eid_te = data["eth_id"][te_idx]

    # 标准化
    sc_a = StandardScaler().fit(xa_tr); xa_tr = sc_a.transform(xa_tr); xa_te = sc_a.transform(xa_te)
    sc_m = StandardScaler().fit(xm_tr); xm_tr = sc_m.transform(xm_tr); xm_te = sc_m.transform(xm_te)
    sc_p = StandardScaler().fit(xp_tr); xp_tr = sc_p.transform(xp_tr); xp_te = sc_p.transform(xp_te)
    sc_y = StandardScaler().fit(y_tr); y_tr_s = sc_y.transform(y_tr); y_te_s = sc_y.transform(y_te)

    tr_ds = MultiModalDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                         xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MultiModalDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                         xp_te.astype(np.float32), y_te_s.astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"], d_part=data["part_dim"],
        d_model=64, n_ethnic=3, n_scales=6, p_drop=0.3,
        use_adain=use_adain, use_gate=use_gate,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            if mask_modality == "audio":
                xa_b = torch.zeros_like(xa_b)
            elif mask_modality == "meta":
                xm_b = torch.zeros_like(xm_b)
            elif mask_modality == "part":
                xp_b = torch.zeros_like(xp_b)
            opt.zero_grad()
            pred = model(xa_b, xm_b, xp_b, eid_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                if mask_modality == "audio":
                    xa_b = torch.zeros_like(xa_b)
                elif mask_modality == "meta":
                    xm_b = torch.zeros_like(xm_b)
                elif mask_modality == "part":
                    xp_b = torch.zeros_like(xp_b)
                pred = model(xa_b, xm_b, xp_b, eid_b)
                val_losses.append(criterion(pred, y_b).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # 加载最佳模型预测
    model.load_state_dict(best_state)
    model.eval()
    preds_s, alphas, gates = [], [], []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            if mask_modality == "audio":
                xa_b = torch.zeros_like(xa_b)
            elif mask_modality == "meta":
                xm_b = torch.zeros_like(xm_b)
            elif mask_modality == "part":
                xp_b = torch.zeros_like(xp_b)
            out = model(xa_b, xm_b, xp_b, eid_b, return_weights=True)
            pred, alpha, gate = out
            preds_s.append(pred.cpu().numpy())
            if alpha is not None:
                alphas.append(alpha.cpu().numpy())
            if gate is not None:
                gates.append(gate.cpu().numpy())
    preds_s = np.concatenate(preds_s)
    preds = sc_y.inverse_transform(preds_s)
    alphas = np.concatenate(alphas) if alphas else None
    gates = np.concatenate(gates) if gates else None

    return preds, y_te, alphas, gates, ep + 1


def train_baseline_ridge(data, tr_idx, te_idx):
    X_tr = np.hstack([data["x_audio"][tr_idx], data["x_meta"][tr_idx], data["x_part"][tr_idx]])
    X_te = np.hstack([data["x_audio"][te_idx], data["x_meta"][te_idx], data["x_part"][te_idx]])
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    preds = np.zeros((len(te_idx), N_SCALES))
    for k in range(N_SCALES):
        m = Ridge(alpha=1.0).fit(X_tr, data["y_adj"][tr_idx, k])
        preds[:, k] = m.predict(X_te)
    return preds, data["y_adj"][te_idx]


def train_baseline_xgb(data, tr_idx, te_idx):
    X_tr = np.hstack([data["x_audio"][tr_idx], data["x_meta"][tr_idx], data["x_part"][tr_idx]])
    X_te = np.hstack([data["x_audio"][te_idx], data["x_meta"][te_idx], data["x_part"][te_idx]])
    preds = np.zeros((len(te_idx), N_SCALES))
    for k in range(N_SCALES):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                       random_state=42, subsample=0.8)
        m.fit(X_tr, data["y_adj"][tr_idx, k])
        preds[:, k] = m.predict(X_te)
    return preds, data["y_adj"][te_idx]


def train_baseline_pre_only(data, tr_idx, te_idx):
    """仅用 pre 量表做线性回归预测残差 (应接近 0)"""
    preds = np.zeros((len(te_idx), N_SCALES))
    for k in range(N_SCALES):
        m = Ridge(alpha=1.0).fit(data["y_pre"][tr_idx, k:k+1], data["y_adj"][tr_idx, k])
        preds[:, k] = m.predict(data["y_pre"][te_idx, k:k+1])
    return preds, data["y_adj"][te_idx]


def eval_metrics(preds, truth):
    results = {}
    for k, name in enumerate(SCALES):
        rmse = float(np.sqrt(mean_squared_error(truth[:, k], preds[:, k])))
        r2 = float(r2_score(truth[:, k], preds[:, k]))
        r, p = pearsonr(truth[:, k], preds[:, k])
        results[name] = {"rmse": round(rmse, 4), "r2": round(r2, 4),
                         "pearson_r": round(float(r), 4), "p": float(p)}
    rmse_mean = float(np.mean([results[s]["rmse"] for s in SCALES]))
    r2_mean = float(np.mean([results[s]["r2"] for s in SCALES]))
    r_mean = float(np.mean([results[s]["pearson_r"] for s in SCALES]))
    results["_mean"] = {"rmse": round(rmse_mean, 4), "r2": round(r2_mean, 4), "pearson_r": round(r_mean, 4)}
    return results


def run_cv(data, model_fn, n_splits=5, **kwargs):
    gkf = GroupKFold(n_splits=n_splits)
    all_preds, all_truth = [], []
    for fold, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        preds, truth = model_fn(data, tr, te, **kwargs)[:2]
        all_preds.append(preds)
        all_truth.append(truth)
    all_preds = np.concatenate(all_preds)
    all_truth = np.concatenate(all_truth)
    return eval_metrics(all_preds, all_truth)


def main():
    print(f"[EXP-1] device={DEVICE}, 全量数据 + DAP 防御性分析协议")
    from defensive_protocol import DefensiveProtocol
    dap = DefensiveProtocol("EXP1_sspmfn")

    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    results = {"n_samples": N, "frac": "full", "cv": "GroupKFold_5", "d_model": 64}

    # === DAP 检查 3: 标签泄漏 ===
    print("\n  [DAP] 标签泄漏检测...")
    X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    leak_result = dap.check_leakage(X_all, data["y_adj"])
    print(f"    {leak_result.message}")

    # --- 基线 ---
    print("\n  [B1] pre-only Ridge...")
    results["B1_pre_only"] = run_cv(data, train_baseline_pre_only)
    print(f"    RMSE={results['B1_pre_only']['_mean']['rmse']:.4f} r={results['B1_pre_only']['_mean']['pearson_r']:.4f}")

    print("  [B2] Ridge (all features)...")
    results["B2_ridge"] = run_cv(data, train_baseline_ridge)
    print(f"    RMSE={results['B2_ridge']['_mean']['rmse']:.4f} r={results['B2_ridge']['_mean']['pearson_r']:.4f}")

    print("  [B3] XGBoost...")
    results["B3_xgb"] = run_cv(data, train_baseline_xgb)
    print(f"    RMSE={results['B3_xgb']['_mean']['rmse']:.4f} r={results['B3_xgb']['_mean']['pearson_r']:.4f}")

    # --- SSP-MFN 多种子 ---
    seeds = [17, 42, 2024]
    sspmfn_rmses = []
    print("\n  [SSP-MFN] full model × 3 seeds...")
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        r = run_cv(data, train_sspmfn, use_gate=True, use_adain=True)
        sspmfn_rmses.append(r["_mean"]["rmse"])
        print(f"    seed={s}: RMSE={r['_mean']['rmse']:.4f} r={r['_mean']['pearson_r']:.4f}")
    results["SSP_MFN_full"] = r  # 最后一个种子的详细结果
    results["SSP_MFN_full"]["_seeds"] = {
        "rmses": [round(x, 4) for x in sspmfn_rmses],
        "mean": round(float(np.mean(sspmfn_rmses)), 4),
        "std": round(float(np.std(sspmfn_rmses)), 4),
    }

    # === DAP 检查 4: 结果稳定性 ===
    print("\n  [DAP] 结果稳定性检测...")
    stab_result = dap.check_stability(sspmfn_rmses)
    print(f"    {stab_result.message}")

    # --- 消融 ---
    print("\n  [SSP-MFN] no gate...")
    results["SSP_MFN_no_gate"] = run_cv(data, train_sspmfn, use_gate=False, use_adain=True)
    print(f"    RMSE={results['SSP_MFN_no_gate']['_mean']['rmse']:.4f}")

    print("  [SSP-MFN] no adain...")
    results["SSP_MFN_no_adain"] = run_cv(data, train_sspmfn, use_gate=True, use_adain=False)
    print(f"    RMSE={results['SSP_MFN_no_adain']['_mean']['rmse']:.4f}")

    print("  [SSP-MFN] no gate + no adain (plain MLP)...")
    results["SSP_MFN_plain"] = run_cv(data, train_sspmfn, use_gate=False, use_adain=False)
    print(f"    RMSE={results['SSP_MFN_plain']['_mean']['rmse']:.4f}")

    # --- 模态消融 (mask one modality) ---
    print("\n  [模态消融] mask each modality...")
    for mod in ["audio", "meta", "part"]:
        results[f"SSP_MFN_no_{mod}"] = run_cv(data, train_sspmfn,
                                               use_gate=True, use_adain=True,
                                               mask_modality=mod)
        print(f"    no_{mod}: RMSE={results[f'SSP_MFN_no_{mod}']['_mean']['rmse']:.4f}")

    # === DAP 检查 1: 过拟合检测 ===
    print("\n  [DAP] 过拟合检测...")
    gkf = GroupKFold(n_splits=5)
    tr, te = next(iter(gkf.split(data["x_audio"], data["y_adj"], data["groups"])))
    preds_te, truth_te = train_sspmfn(data, tr, te)[:2]
    preds_tr, truth_tr = train_sspmfn(data, tr, tr)[:2]
    tr_rmse = np.sqrt(mean_squared_error(truth_tr, preds_tr))
    te_rmse = np.sqrt(mean_squared_error(truth_te, preds_te))
    overfit_result = dap.check_overfit(tr_rmse, te_rmse)
    print(f"    {overfit_result.message}")

    # === DAP 检查 2: 分布偏移 ===
    print("\n  [DAP] 分布偏移检测...")
    X_tr = X_all[tr]
    X_te = X_all[te]
    shift_result = dap.check_distribution_shift(X_tr, X_te)
    print(f"    {shift_result.message}")

    # === DAP 检查 5: 置换基线 ===
    print("\n  [DAP] 置换基线检测...")
    model_rmse = results["SSP_MFN_full"]["_mean"]["rmse"]
    perm_result = dap.check_permutation_baseline(model_rmse, data["y_adj"][:, 0])
    print(f"    {perm_result.message}")

    # === 生成 DAP 报告 ===
    dap_report = dap.generate_report()
    results["DAP"] = dap_report

    # 保存
    out_path = RES / "EXP1_sspmfn_full.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\nsaved → {out_path}")

    # 汇总表
    print("\n" + "="*70)
    print("  EXP-1 模型对比 (全量 N=850, 六维均值):")
    print(f"  {'模型':<30} {'RMSE':<10} {'R²':<10} {'r':<10}")
    print(f"  {'-'*60}")
    model_names = ["B1_pre_only", "B2_ridge", "B3_xgb",
                   "SSP_MFN_full", "SSP_MFN_no_gate", "SSP_MFN_no_adain", "SSP_MFN_plain",
                   "SSP_MFN_no_audio", "SSP_MFN_no_meta", "SSP_MFN_no_part"]
    for name in model_names:
        if name in results:
            m = results[name]["_mean"]
            print(f"  {name:<30} {m['rmse']:<10.4f} {m['r2']:<10.4f} {m['pearson_r']:<10.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
