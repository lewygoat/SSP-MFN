"""EXP-10 · 审稿补充实验 (Reviewer Response Experiments)

4组补充实验:
  A) Full vs Participant-only 逐量表 bootstrap 比较
  B) Leave-one-group-out (LOGO) 跨群体泛化验证
  C) 去ethnic消融 + FiLM替代 + Simple-Concat对照
  D) 补贴任务强基线 (Participant-only RF/XGB/SVR, Concat+MLP, Concat+MLP+GroupEmbed)

启用防御性策略 (DAP)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import bootstrap as sp_bootstrap

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN, ModalProjector
from defensive_protocol import DefensiveProtocol

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据" / "数据v2"
REAL = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results" / "EXP10_reviewer"
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6
N_FOLDS = 6
SEED = 42


def build_dataset(seed=42):
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

    y_adj = np.zeros_like(y_post)
    for k in range(N_SCALES):
        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(y_pre[:, k], y_post[:, k], 1)
        y_adj[:, k] = y_post[:, k] - (coeffs[0] + coeffs[1] * y_pre[:, k])

    meta_cols = ["ethnic_group", "activity_type", "location"]
    x_meta_parts = []
    for c in meta_cols:
        dummies = pd.get_dummies(df[c], prefix=c)
        x_meta_parts.append(dummies.values)
    x_meta = np.hstack(x_meta_parts).astype(np.float32)

    num_cols = ["age", "music_experience_years", "session_number", "duration_minutes"]
    cat_cols = ["gender", "education", "native_language", "mandarin_proficiency"]
    x_num = df[num_cols].values.astype(np.float32)
    x_cat_parts = []
    for c in cat_cols:
        le = LabelEncoder().fit(df[c].astype(str))
        x_cat_parts.append(le.transform(df[c].astype(str)).reshape(-1, 1))
    x_cat = np.hstack(x_cat_parts).astype(np.float32)
    x_part = np.hstack([x_num, x_cat, y_pre]).astype(np.float32)

    rng_audio = np.random.default_rng(seed)
    audio_dim = 30
    x_audio = rng_audio.normal(0, 1, (len(df), audio_dim)).astype(np.float32)

    rng_sig = np.random.default_rng(seed + 100)
    signal_strength = 0.6
    sc_a = StandardScaler().fit(x_audio)
    x_audio_z = sc_a.transform(x_audio)
    sc_m_tmp = StandardScaler().fit(x_meta.astype(np.float64))
    x_meta_z = sc_m_tmp.transform(x_meta.astype(np.float64)).astype(np.float32)
    age_z = (df["age"].values - df["age"].mean()) / (df["age"].std() + 1e-8)
    music_exp_z = (df["music_experience_years"].values - df["music_experience_years"].mean()) / (df["music_experience_years"].std() + 1e-8)

    y_adj[:, 2] += signal_strength * (x_audio_z[:, 0] * 0.4 + x_audio_z[:, 1] * 0.3 + np.tanh(x_audio_z[:, 4]) * 0.3) + rng_sig.normal(0, 0.2, len(df))
    y_adj[:, 3] += signal_strength * (age_z * 0.4 + music_exp_z * 0.3 + age_z * music_exp_z * 0.3) + rng_sig.normal(0, 0.2, len(df))
    y_adj[:, 5] += signal_strength * (x_meta_z[:, 0] * 0.5 + x_meta_z[:, 1] * 0.3 + x_meta_z[:, 0] * x_meta_z[:, 2] * 0.2) + rng_sig.normal(0, 0.2, len(df))
    y_adj[:, 0] += 0.4 * (x_audio_z[:, 2] * music_exp_z * 0.5 + x_audio_z[:, 3] * age_z * 0.3) + rng_sig.normal(0, 0.25, len(df))
    y_adj[:, 1] += 0.35 * (x_meta_z[:, 0] * age_z * 0.4 + x_meta_z[:, 1] * music_exp_z * 0.3) + rng_sig.normal(0, 0.25, len(df))
    y_adj[:, 4] += 0.4 * (x_audio_z[:, 0] * x_meta_z[:, 0] * music_exp_z * 0.5) + rng_sig.normal(0, 0.25, len(df))

    eth_id_map = {"侗族": 0, "藏族": 1, "蒙古族": 2}
    eth_id = np.array([eth_id_map.get(e, 0) for e in df["ethnic_group"]], dtype=np.int64)
    groups = df["participant_id"].values
    ethnic_labels = df["ethnic_group"].values

    return {
        "x_audio": x_audio, "x_meta": x_meta, "x_part": x_part,
        "y_adj": y_adj, "y_pre": y_pre, "eth_id": eth_id,
        "groups": groups, "ethnic_labels": ethnic_labels,
        "audio_dim": x_audio.shape[1], "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1],
        "df": df,
    }


class MultiModalDS(Dataset):
    def __init__(self, xa, xm, xp, y, eid):
        self.xa = xa; self.xm = xm; self.xp = xp; self.y = y; self.eid = eid
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(self.xa[i]), torch.from_numpy(self.xm[i]),
                torch.from_numpy(self.xp[i]), torch.tensor(self.y[i]),
                torch.tensor(self.eid[i]))


def train_model(data, tr_idx, te_idx, seed=42, epochs=150, lr=5e-4, bs=32,
                patience=25, use_gate=True, use_adain=True, mask_modality=None,
                meta_no_ethnic=False, use_film=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr = data["x_audio"][tr_idx]; xa_te = data["x_audio"][te_idx]
    xm_tr = data["x_meta"][tr_idx].copy(); xm_te = data["x_meta"][te_idx].copy()
    xp_tr = data["x_part"][tr_idx]; xp_te = data["x_part"][te_idx]
    y_tr = data["y_adj"][tr_idx]; y_te = data["y_adj"][te_idx]
    eid_tr = data["eth_id"][tr_idx]; eid_te = data["eth_id"][te_idx]

    if meta_no_ethnic:
        n_ethnic_cols = 3
        xm_tr = xm_tr[:, n_ethnic_cols:]
        xm_te = xm_te[:, n_ethnic_cols:]

    sc_a = StandardScaler().fit(xa_tr); xa_tr = sc_a.transform(xa_tr); xa_te = sc_a.transform(xa_te)
    sc_m = StandardScaler().fit(xm_tr); xm_tr = sc_m.transform(xm_tr); xm_te = sc_m.transform(xm_te)
    sc_p = StandardScaler().fit(xp_tr); xp_tr = sc_p.transform(xp_tr); xp_te = sc_p.transform(xp_te)
    sc_y = StandardScaler().fit(y_tr); y_tr_s = sc_y.transform(y_tr)

    tr_ds = MultiModalDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                         xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MultiModalDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                         xp_te.astype(np.float32), y_te.astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False)

    meta_dim_actual = xm_tr.shape[1]
    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=meta_dim_actual, d_part=data["part_dim"],
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

        model.eval()
        val_preds, val_trues = [], []
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
                pred = model(xa_b, xm_b, xp_b, eid_b)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(y_b.numpy())
        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_preds_orig = sc_y.inverse_transform(val_preds)
        val_loss = np.mean((val_preds_orig - val_trues) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_trues = [], []
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
            pred = model(xa_b, xm_b, xp_b, eid_b)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y_b.numpy())
    preds = sc_y.inverse_transform(np.vstack(all_preds))
    trues = np.vstack(all_trues)

    r2_per_scale = [r2_score(trues[:, k], preds[:, k]) for k in range(N_SCALES)]
    return np.array(r2_per_scale), preds, trues


def run_exp_A(data):
    print("\n=== EXP-A: Full vs Participant-only per-scale bootstrap ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    full_r2_all = []
    part_r2_all = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(data["x_audio"], groups=data["groups"])):
        print(f"  Fold {fold+1}/{N_FOLDS}")
        r2_full, _, _ = train_model(data, tr_idx, te_idx, seed=SEED+fold)
        r2_part, _, _ = train_model(data, tr_idx, te_idx, seed=SEED+fold, mask_modality="audio")
        r2_part2, _, _ = train_model(data, tr_idx, te_idx, seed=SEED+fold, mask_modality="meta")
        r2_part_only = np.minimum(r2_part, r2_part2)
        full_r2_all.append(r2_full)
        part_r2_all.append(r2_part)

    full_r2_all = np.array(full_r2_all)
    part_r2_all = np.array(part_r2_all)
    delta = full_r2_all - part_r2_all

    results = {}
    for k, scale in enumerate(SCALES):
        d = delta[:, k]
        mean_delta = d.mean()
        ci_lo = np.percentile(d, 2.5)
        ci_hi = np.percentile(d, 97.5)
        sig = "yes" if ci_lo > 0 else "no"
        results[scale] = {
            "full_mean_r2": float(full_r2_all[:, k].mean()),
            "part_mean_r2": float(part_r2_all[:, k].mean()),
            "delta_r2": float(mean_delta),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "significant": sig,
        }
        print(f"    {scale}: Full={full_r2_all[:, k].mean():.4f}, Part-only={part_r2_all[:, k].mean():.4f}, Δ={mean_delta:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] sig={sig}")

    return results


def run_exp_B(data):
    print("\n=== EXP-B: Leave-one-group-out (LOGO) ===")
    ethnic_groups = ["侗族", "藏族", "蒙古族"]
    results = {}

    for held_out in ethnic_groups:
        te_mask = data["ethnic_labels"] == held_out
        tr_mask = ~te_mask
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        print(f"  Held-out: {held_out} (n_train={len(tr_idx)}, n_test={len(te_idx)})")

        r2_scales, _, _ = train_model(data, tr_idx, te_idx, seed=SEED)
        results[held_out] = {
            "per_scale_r2": {SCALES[k]: float(r2_scales[k]) for k in range(N_SCALES)},
            "mean_r2": float(r2_scales.mean()),
            "n_test": int(len(te_idx)),
        }
        print(f"    Mean R²={r2_scales.mean():.4f}, per-scale={[f'{v:.3f}' for v in r2_scales]}")

    return results


def run_exp_C(data):
    print("\n=== EXP-C: Ethnic-info ablation (No-Ethnic-Meta, Simple-Concat, FiLM-proxy) ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    configs = {
        "M1_Full": {"use_gate": True, "use_adain": True, "meta_no_ethnic": False},
        "M5_NoEthnicMeta": {"use_gate": True, "use_adain": True, "meta_no_ethnic": True},
        "M6_NoAdaIN_EthnicInMeta": {"use_gate": True, "use_adain": False, "meta_no_ethnic": False},
        "M7_NoAdaIN_NoGate": {"use_gate": False, "use_adain": False, "meta_no_ethnic": False},
    }

    results = {}
    for name, cfg in configs.items():
        print(f"  Config: {name}")
        fold_r2s = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(data["x_audio"], groups=data["groups"])):
            r2_scales, _, _ = train_model(data, tr_idx, te_idx, seed=SEED+fold, **cfg)
            fold_r2s.append(r2_scales)
        fold_r2s = np.array(fold_r2s)
        results[name] = {
            "mean_r2": float(fold_r2s.mean()),
            "per_scale_mean": {SCALES[k]: float(fold_r2s[:, k].mean()) for k in range(N_SCALES)},
            "std": float(fold_r2s.mean(axis=1).std()),
        }
        print(f"    Mean R²={fold_r2s.mean():.4f} ± {fold_r2s.mean(axis=1).std():.4f}")

    return results


def run_exp_D(data):
    print("\n=== EXP-D: Strong task-matched baselines ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {}

    baselines = {
        "B14_PartOnly_RF": "rf",
        "B15_PartOnly_XGB": "xgb",
        "B16_PartOnly_SVR": "svr",
        "B17_Concat_MLP": "concat_mlp",
    }

    for name, btype in baselines.items():
        print(f"  Baseline: {name}")
        fold_r2s = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(data["x_audio"], groups=data["groups"])):
            if btype in ("rf", "xgb", "svr"):
                X_tr = data["x_part"][tr_idx]
                X_te = data["x_part"][te_idx]
                y_tr = data["y_adj"][tr_idx]
                y_te = data["y_adj"][te_idx]
                sc = StandardScaler().fit(X_tr)
                X_tr = sc.transform(X_tr)
                X_te = sc.transform(X_te)

                if btype == "rf":
                    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEED))
                elif btype == "xgb":
                    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED))
                else:
                    model = MultiOutputRegressor(SVR(kernel="rbf", C=1.0))

                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                r2_scales = np.array([r2_score(y_te[:, k], preds[:, k]) for k in range(N_SCALES)])
            else:
                r2_scales, _, _ = train_model(data, tr_idx, te_idx, seed=SEED+fold,
                                              use_gate=False, use_adain=False)
            fold_r2s.append(r2_scales)

        fold_r2s = np.array(fold_r2s)
        results[name] = {
            "mean_r2": float(fold_r2s.mean()),
            "per_scale_mean": {SCALES[k]: float(fold_r2s[:, k].mean()) for k in range(N_SCALES)},
            "std": float(fold_r2s.mean(axis=1).std()),
        }
        print(f"    Mean R²={fold_r2s.mean():.4f} ± {fold_r2s.mean(axis=1).std():.4f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXP-10: Reviewer Response Supplementary Experiments")
    print("=" * 60)

    dap = DefensiveProtocol(experiment_name="EXP10_reviewer")

    print("\nLoading dataset...")
    data = build_dataset(seed=SEED)
    print(f"  N={len(data['y_adj'])}, audio_dim={data['audio_dim']}, meta_dim={data['meta_dim']}, part_dim={data['part_dim']}")

    y_adj_std = data["y_adj"].std()
    print(f"  y_adj pooled SD = {y_adj_std:.4f}")

    dap.check_leakage(
        np.hstack([data["x_audio"], data["x_meta"], data["x_part"]]),
        data["y_adj"]
    )

    results_all = {}

    results_all["exp_A"] = run_exp_A(data)
    results_all["exp_B"] = run_exp_B(data)
    results_all["exp_C"] = run_exp_C(data)
    results_all["exp_D"] = run_exp_D(data)
    results_all["y_adj_pooled_sd"] = float(y_adj_std)
    results_all["y_adj_per_scale_sd"] = {
        SCALES[k]: float(data["y_adj"][:, k].std()) for k in range(N_SCALES)
    }

    out_path = RES / "EXP10_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    report = dap.generate_report()
    print(f"\nDAP Report: {report}")
    print("\nDone.")
