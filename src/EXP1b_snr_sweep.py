"""EXP-1b · SNR 敏感性分析 (Sensitivity Analysis)

6 组 SNR × 2 组交互占比 × 2 组样本量 = 12 实验点
每点跑: SSP-MFN / Ridge / XGBoost / pre-only
集成 DAP 防御性分析协议

输出: 实验/results/EXP1b_snr_sweep.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge as RidgeModel
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据" / "数据v2"
REAL = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6

# === 实验矩阵 ===
SWEEP_CONFIGS = [
    {"name": "S1_lowSNR_lowInt",  "signal": 0.5, "noise": 1.0, "interact_ratio": 0.3},
    {"name": "S2_lowSNR_highInt", "signal": 0.5, "noise": 1.0, "interact_ratio": 0.7},
    {"name": "S3_midSNR_lowInt",  "signal": 1.0, "noise": 0.6, "interact_ratio": 0.3},
    {"name": "S4_midSNR_highInt", "signal": 1.0, "noise": 0.6, "interact_ratio": 0.7},
    {"name": "S5_highSNR_lowInt", "signal": 1.5, "noise": 0.3, "interact_ratio": 0.3},
    {"name": "S6_highSNR_highInt","signal": 1.5, "noise": 0.3, "interact_ratio": 0.7},
]


def build_base_features(seed=42):
    """构建三路特征(不含标签), 只跑一次"""
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

    y_pre = df[[f"{d}_pre" for d in SCALES]].values.astype(np.float32)
    y_post = df[[f"{d}_post" for d in SCALES]].values.astype(np.float32)
    y_adj_base = np.zeros_like(y_post)
    for k in range(N_SCALES):
        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(y_pre[:, k], y_post[:, k], 1)
        y_adj_base[:, k] = y_post[:, k] - (coeffs[0] + coeffs[1] * y_pre[:, k])

    # 音频
    import librosa
    manifest = pd.read_parquet(REAL / "clips_30s_manifest.parquet")
    sub = manifest.groupby("ethnic_group", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 20), random_state=seed)).reset_index(drop=True)
    audio_feats_by_ethnic = {}
    for _, r in sub.iterrows():
        p = REAL / r["out_path"]
        try:
            y_wav, sr = librosa.load(str(p), sr=22050, mono=True, duration=30.0)
            mfcc = librosa.feature.mfcc(y=y_wav, sr=sr, n_mfcc=13).mean(axis=1)
            sc = librosa.feature.spectral_centroid(y=y_wav, sr=sr).mean()
            sb = librosa.feature.spectral_bandwidth(y=y_wav, sr=sr).mean()
            sr2 = librosa.feature.spectral_rolloff(y=y_wav, sr=sr).mean()
            zcr = librosa.feature.zero_crossing_rate(y_wav).mean()
            tempo, _ = librosa.beat.beat_track(y=y_wav, sr=sr)
            tv = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            chroma = librosa.feature.chroma_stft(y=y_wav, sr=sr).mean(axis=1)
            feat = np.concatenate([mfcc, [sc, sb, sr2, zcr, tv], chroma])
            eg = r["ethnic_group"]
            audio_feats_by_ethnic.setdefault(eg, []).append(feat)
        except:
            pass
    eth2audio = {k: np.stack(v).mean(0) for k, v in audio_feats_by_ethnic.items()}
    audio_dim = len(next(iter(eth2audio.values())))
    eth_map = {"侗族": "dong", "藏族": "tibetan", "蒙古族": "mongolian"}
    rng_n = np.random.default_rng(seed + 1)
    x_audio = np.stack([eth2audio.get(eth_map.get(e, ""), np.zeros(audio_dim))
                        for e in df["ethnic_group"]]).astype(np.float32)
    x_audio += rng_n.normal(0, 0.1, x_audio.shape).astype(np.float32)

    # 文化元
    x_meta = np.hstack([pd.get_dummies(df[c], prefix=c).values
                        for c in ["ethnic_group", "activity_type", "location"]]).astype(np.float32)
    # 参与者
    x_num = df[["age", "music_experience_years", "session_number", "duration_minutes"]].values.astype(np.float32)
    x_cat = np.hstack([LabelEncoder().fit_transform(df[c].astype(str)).reshape(-1, 1)
                       for c in ["gender", "education", "native_language", "mandarin_proficiency"]]).astype(np.float32)
    x_part = np.hstack([x_num, x_cat, y_pre]).astype(np.float32)

    eth_id = np.array([{"侗族": 0, "藏族": 1, "蒙古族": 2}[e] for e in df["ethnic_group"]], dtype=np.int64)
    groups = df["participant_id"].values

    return {"x_audio": x_audio, "x_meta": x_meta, "x_part": x_part,
            "y_adj_base": y_adj_base, "eth_id": eth_id, "groups": groups,
            "df": df, "audio_dim": x_audio.shape[1],
            "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1]}


def plant_signal(base, signal_strength, noise_level, interact_ratio, seed=42):
    """在基础残差上植入可控信号"""
    rng = np.random.default_rng(seed + 200)
    y_adj = base["y_adj_base"].copy()
    df = base["df"]
    N = len(df)

    x_audio_z = StandardScaler().fit_transform(base["x_audio"])
    x_meta_z = StandardScaler().fit_transform(base["x_meta"].astype(np.float64)).astype(np.float32)
    age_z = (df["age"].values - df["age"].mean()) / (df["age"].std() + 1e-8)
    mexp_z = (df["music_experience_years"].values - df["music_experience_years"].mean()) / (df["music_experience_years"].std() + 1e-8)

    S = signal_strength
    noise = noise_level
    lin = 1.0 - interact_ratio  # 线性信号占比
    inter = interact_ratio       # 交互信号占比

    # CSAS (idx=2): 音频主导
    lin_sig = x_audio_z[:, 0] * 0.4 + x_audio_z[:, 1] * 0.3 + x_audio_z[:, 4] * 0.3
    int_sig = x_audio_z[:, 0] * mexp_z * 0.5 + x_audio_z[:, 2] * age_z * 0.3 + np.tanh(x_audio_z[:, 3] * mexp_z) * 0.2
    y_adj[:, 2] += S * (lin * lin_sig + inter * int_sig) + rng.normal(0, noise, N)

    # SSCS (idx=3): 参与者主导
    lin_sig = age_z * 0.5 + mexp_z * 0.3 + (age_z > 0).astype(float) * 0.2
    int_sig = age_z * mexp_z * 0.5 + np.tanh(age_z * 2) * mexp_z * 0.3 + age_z * x_audio_z[:, 0] * 0.2
    y_adj[:, 3] += S * (lin * lin_sig + inter * int_sig) + rng.normal(0, noise, N)

    # SCI2 (idx=5): 文化元主导
    lin_sig = x_meta_z[:, 0] * 0.5 + x_meta_z[:, 1] * 0.3 + x_meta_z[:, 2] * 0.2
    int_sig = x_meta_z[:, 0] * age_z * 0.4 + x_meta_z[:, 1] * mexp_z * 0.3 + x_meta_z[:, 0] * x_audio_z[:, 0] * 0.3
    y_adj[:, 5] += S * (lin * lin_sig + inter * int_sig) + rng.normal(0, noise, N)

    # ICS (idx=0): 跨模态 audio×part
    int_sig = x_audio_z[:, 2] * mexp_z * 0.5 + x_audio_z[:, 3] * age_z * 0.3 + x_audio_z[:, 1] * x_meta_z[:, 0] * 0.2
    y_adj[:, 0] += S * 0.7 * (lin * x_audio_z[:, 0] * 0.5 + inter * int_sig) + rng.normal(0, noise, N)

    # IRI (idx=1): 跨模态 meta×part
    int_sig = x_meta_z[:, 0] * age_z * 0.4 + x_meta_z[:, 1] * mexp_z * 0.3 + x_meta_z[:, 2] * x_audio_z[:, 1] * 0.3
    y_adj[:, 1] += S * 0.6 * (lin * age_z * 0.5 + inter * int_sig) + rng.normal(0, noise, N)

    # IOS (idx=4): 三模态交互
    int_sig = x_audio_z[:, 0] * x_meta_z[:, 0] * mexp_z * 0.6 + x_audio_z[:, 1] * age_z * x_meta_z[:, 1] * 0.4
    y_adj[:, 4] += S * 0.6 * (lin * x_meta_z[:, 0] * 0.5 + inter * int_sig) + rng.normal(0, noise, N)

    return y_adj


class MMDS(Dataset):
    def __init__(self, xa, xm, xp, y, eid):
        self.xa, self.xm, self.xp = torch.FloatTensor(xa), torch.FloatTensor(xm), torch.FloatTensor(xp)
        self.y = torch.FloatTensor(y)
        self.eid = torch.LongTensor(eid)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.xa[i], self.xm[i], self.xp[i], self.y[i], self.eid[i]


def train_sspmfn_quick(data, tr, te, y_adj):
    """快速版 SSP-MFN 训练 (用于 sweep)"""
    xa, xm, xp = data["x_audio"], data["x_meta"], data["x_part"]
    eid = data["eth_id"]
    sc_a = StandardScaler().fit(xa[tr]); xa_tr = sc_a.transform(xa[tr]); xa_te = sc_a.transform(xa[te])
    sc_m = StandardScaler().fit(xm[tr]); xm_tr = sc_m.transform(xm[tr]); xm_te = sc_m.transform(xm[te])
    sc_p = StandardScaler().fit(xp[tr]); xp_tr = sc_p.transform(xp[tr]); xp_te = sc_p.transform(xp[te])
    sc_y = StandardScaler().fit(y_adj[tr]); y_tr = sc_y.transform(y_adj[tr])

    tr_dl = DataLoader(MMDS(xa_tr, xm_tr, xp_tr, y_tr, eid[tr]), batch_size=32, shuffle=True)
    model = SSPMFN(d_audio=data["audio_dim"], d_meta=data["meta_dim"], d_part=data["part_dim"],
                   d_model=64, n_ethnic=3, n_scales=6, p_drop=0.3, use_adain=True, use_gate=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    best_state, best_loss = None, float("inf")
    patience_cnt = 0

    for ep in range(150):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xa_b, xm_b, xp_b, eid_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # 简单验证
        model.eval()
        with torch.no_grad():
            xa_t = torch.FloatTensor(xa_te).to(DEVICE)
            xm_t = torch.FloatTensor(xm_te).to(DEVICE)
            xp_t = torch.FloatTensor(xp_te).to(DEVICE)
            eid_t = torch.LongTensor(eid[te]).to(DEVICE)
            val_loss = criterion(model(xa_t, xm_t, xp_t, eid_t),
                                 torch.FloatTensor(sc_y.transform(y_adj[te])).to(DEVICE)).item()
        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= 20:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.FloatTensor(xa_te).to(DEVICE),
                       torch.FloatTensor(xm_te).to(DEVICE),
                       torch.FloatTensor(xp_te).to(DEVICE),
                       torch.LongTensor(eid[te]).to(DEVICE)).cpu().numpy()
    return sc_y.inverse_transform(pred_s)


def train_ridge_quick(data, tr, te, y_adj):
    X_tr = np.hstack([data["x_audio"][tr], data["x_meta"][tr], data["x_part"][tr]])
    X_te = np.hstack([data["x_audio"][te], data["x_meta"][te], data["x_part"][te]])
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    preds = np.zeros((len(te), N_SCALES))
    for k in range(N_SCALES):
        m = RidgeModel(alpha=1.0).fit(X_tr, y_adj[tr, k])
        preds[:, k] = m.predict(X_te)
    return preds


def train_xgb_quick(data, tr, te, y_adj):
    X_tr = np.hstack([data["x_audio"][tr], data["x_meta"][tr], data["x_part"][tr]])
    X_te = np.hstack([data["x_audio"][te], data["x_meta"][te], data["x_part"][te]])
    preds = np.zeros((len(te), N_SCALES))
    for k in range(N_SCALES):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X_tr, y_adj[tr, k])
        preds[:, k] = m.predict(X_te)
    return preds


def run_one_config(base, config, sample_sizes=[850, 220]):
    """跑一组 SNR 配置"""
    results = {"config": config}
    y_adj = plant_signal(base, config["signal"], config["noise"], config["interact_ratio"])

    for n_sample in sample_sizes:
        key = f"N{n_sample}"
        if n_sample < len(y_adj):
            rng = np.random.default_rng(42)
            pids = np.unique(base["groups"])
            keep = rng.choice(pids, size=int(len(pids) * n_sample / 850), replace=False)
            mask = np.isin(base["groups"], keep)
            idx_all = np.where(mask)[0]
        else:
            idx_all = np.arange(len(y_adj))

        data_sub = {k: v[idx_all] if isinstance(v, np.ndarray) and len(v) == len(y_adj) else v
                    for k, v in base.items()}
        data_sub["audio_dim"] = base["audio_dim"]
        data_sub["meta_dim"] = base["meta_dim"]
        data_sub["part_dim"] = base["part_dim"]
        y_sub = y_adj[idx_all]
        groups_sub = base["groups"][idx_all]

        gkf = GroupKFold(n_splits=5)
        preds_mfn, preds_ridge, preds_xgb = [], [], []
        truths = []

        for fold, (tr, te) in enumerate(gkf.split(data_sub["x_audio"], y_sub, groups_sub)):
            p_mfn = train_sspmfn_quick(data_sub, tr, te, y_sub)
            p_ridge = train_ridge_quick(data_sub, tr, te, y_sub)
            p_xgb = train_xgb_quick(data_sub, tr, te, y_sub)
            preds_mfn.append(p_mfn)
            preds_ridge.append(p_ridge)
            preds_xgb.append(p_xgb)
            truths.append(y_sub[te])

        preds_mfn = np.concatenate(preds_mfn)
        preds_ridge = np.concatenate(preds_ridge)
        preds_xgb = np.concatenate(preds_xgb)
        truths = np.concatenate(truths)

        def metrics(p, t):
            rmse = float(np.sqrt(mean_squared_error(t, p)))
            r2 = float(r2_score(t, p))
            rs = [pearsonr(p[:, k], t[:, k])[0] for k in range(N_SCALES)]
            return {"rmse": round(rmse, 4), "r2": round(r2, 4), "r_mean": round(float(np.mean(rs)), 4)}

        results[key] = {
            "n": len(idx_all),
            "SSP_MFN": metrics(preds_mfn, truths),
            "Ridge": metrics(preds_ridge, truths),
            "XGBoost": metrics(preds_xgb, truths),
            "MFN_wins_Ridge": bool(metrics(preds_mfn, truths)["rmse"] < metrics(preds_ridge, truths)["rmse"]),
        }
    return results


def main():
    print(f"[EXP-1b] SNR 敏感性分析, device={DEVICE}")
    print(f"  6 configs × 2 sample sizes = 12 实验点\n")

    dap = DefensiveProtocol("EXP1b_snr_sweep")

    print("  构建基础特征...")
    base = build_base_features(seed=42)
    print(f"  N={len(base['y_adj_base'])}, audio={base['audio_dim']}, meta={base['meta_dim']}, part={base['part_dim']}")

    all_results = []
    mfn_rmses_850 = []

    for i, cfg in enumerate(SWEEP_CONFIGS):
        print(f"\n  [{i+1}/6] {cfg['name']} (S={cfg['signal']}, σ={cfg['noise']}, int={cfg['interact_ratio']:.0%})")
        res = run_one_config(base, cfg)
        all_results.append(res)

        r850 = res["N850"]
        r220 = res["N220"]
        mfn_rmses_850.append(r850["SSP_MFN"]["rmse"])
        win = "✓ MFN赢" if r850["MFN_wins_Ridge"] else "✗ Ridge赢"
        print(f"    N=850: MFN R²={r850['SSP_MFN']['r2']:.4f} | Ridge R²={r850['Ridge']['r2']:.4f} | {win}")
        print(f"    N=220: MFN R²={r220['SSP_MFN']['r2']:.4f} | Ridge R²={r220['Ridge']['r2']:.4f}")

    # === DAP 检查 ===
    print("\n  [DAP] 稳定性检测 (跨 SNR 配置)...")
    dap.check_stability(mfn_rmses_850)

    # 置换基线 (用中等 SNR 的结果)
    y_mid = plant_signal(base, 1.0, 0.6, 0.5)
    dap.check_permutation_baseline(all_results[2]["N850"]["SSP_MFN"]["rmse"], y_mid[:, 0])

    dap_report = dap.generate_report()

    # === 汇总表 ===
    print("\n" + "="*80)
    print("  SNR 敏感性分析汇总 (N=850)")
    print(f"  {'配置':<25} {'S':>4} {'σ':>4} {'Int%':>5} {'MFN_R²':>8} {'Ridge_R²':>9} {'XGB_R²':>8} {'胜者':>10}")
    print(f"  {'-'*75}")
    for res in all_results:
        c = res["config"]
        r = res["N850"]
        winner = "SSP-MFN" if r["MFN_wins_Ridge"] else "Ridge"
        print(f"  {c['name']:<25} {c['signal']:>4.1f} {c['noise']:>4.1f} {c['interact_ratio']:>5.0%}"
              f" {r['SSP_MFN']['r2']:>8.4f} {r['Ridge']['r2']:>9.4f} {r['XGBoost']['r2']:>8.4f} {winner:>10}")

    print(f"\n  {'配置':<25} {'N=850 MFN赢?':>14} {'N=220 MFN赢?':>14}")
    print(f"  {'-'*55}")
    for res in all_results:
        c = res["config"]
        w850 = "✓" if res["N850"]["MFN_wins_Ridge"] else "✗"
        w220 = "✓" if res["N220"]["MFN_wins_Ridge"] else "✗"
        print(f"  {c['name']:<25} {w850:>14} {w220:>14}")
    print("="*80)

    # 保存
    out = {"sweep": all_results, "DAP": dap_report}
    out_path = RES / "EXP1b_snr_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
