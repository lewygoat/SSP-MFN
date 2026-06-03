"""EXP-8 · 补充对比实验: 多模态融合 SOTA + 领域自适应基线

新增基线:
  B9_MulT       — Multimodal Transformer (cross-modal attention)
  B10_TFN       — Tensor Fusion Network (outer product)
  B11_LMF       — Low-rank Multimodal Fusion
  B12_MLP_MMD   — MLP + MMD 领域自适应
  B13_MLP_CORAL — MLP + CORAL 领域自适应

对比目的:
  1. 证明 SSP-MFN 在小样本下优于大参数量 SOTA (MulT)
  2. 证明门控融合优于张量融合 (TFN/LMF)
  3. 证明 AdaIN 优于传统领域自适应 (MMD/CORAL)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN
from baselines_multimodal import MulT, TFN, LMF
from baselines_domain_adapt import MLPWithMMD, MLPWithCORAL
from defensive_protocol import DefensiveProtocol

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"; RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]
N_SCALES = 6


# 复用 EXP-1 的 build_dataset
from EXP1_sspmfn_main import build_dataset, MultiModalDS


def train_neural_baseline(data, tr_idx, te_idx, model_class, model_kwargs,
                          seed=42, epochs=200, lr=5e-4, bs=32, patience=30,
                          use_domain_loss=False):
    """通用神经网络基线训练函数"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr = data["x_audio"][tr_idx]; xa_te = data["x_audio"][te_idx]
    xm_tr = data["x_meta"][tr_idx]; xm_te = data["x_meta"][te_idx]
    xp_tr = data["x_part"][tr_idx]; xp_te = data["x_part"][te_idx]
    y_tr = data["y_adj"][tr_idx]; y_te = data["y_adj"][te_idx]
    eid_tr = data["eth_id"][tr_idx]; eid_te = data["eth_id"][te_idx]

    sc_a = StandardScaler().fit(xa_tr); xa_tr = sc_a.transform(xa_tr); xa_te = sc_a.transform(xa_te)
    sc_m = StandardScaler().fit(xm_tr); xm_tr = sc_m.transform(xm_tr); xm_te = sc_m.transform(xm_te)
    sc_p = StandardScaler().fit(xp_tr); xp_tr = sc_p.transform(xp_tr); xp_te = sc_p.transform(xp_te)
    sc_y = StandardScaler().fit(y_tr); y_tr_s = sc_y.transform(y_tr)

    tr_ds = MultiModalDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                         xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MultiModalDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                         xp_te.astype(np.float32), sc_y.transform(y_te).astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False)

    model = model_class(**model_kwargs).to(DEVICE)
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
            opt.zero_grad()
            if use_domain_loss and hasattr(model, 'compute_loss'):
                loss = model.compute_loss(xa_b, xm_b, xp_b, y_b, eid_b, criterion)
            else:
                pred = model(xa_b, xm_b, xp_b, eid_b)
                loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
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

    model.load_state_dict(best_state)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            preds_list.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    preds_s = np.concatenate(preds_list)
    preds = sc_y.inverse_transform(preds_s)
    return preds, y_te, model


def eval_metrics(preds, truth):
    """计算每个量表的指标"""
    results = {}
    for k, name in enumerate(SCALES):
        rmse = np.sqrt(mean_squared_error(truth[:, k], preds[:, k]))
        r2 = r2_score(truth[:, k], preds[:, k])
        r, _ = pearsonr(truth[:, k], preds[:, k])
        results[name] = {"rmse": round(rmse, 4), "r2": round(r2, 4), "r": round(float(r), 4)}
    results["_mean"] = {
        "rmse": round(np.mean([results[s]["rmse"] for s in SCALES]), 4),
        "r2": round(np.mean([results[s]["r2"] for s in SCALES]), 4),
        "r": round(np.mean([results[s]["r"] for s in SCALES]), 4),
    }
    return results


def run_cv_neural(data, model_class, model_kwargs, n_splits=5, seed=42,
                  use_domain_loss=False, lr=5e-4):
    """5-fold CV for neural baselines"""
    gkf = GroupKFold(n_splits=n_splits)
    all_preds, all_truth = [], []
    for fold, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        p, t, _ = train_neural_baseline(data, tr, te, model_class, model_kwargs,
                                        seed=seed+fold, use_domain_loss=use_domain_loss, lr=lr)
        all_preds.append(p)
        all_truth.append(t)
    all_preds = np.concatenate(all_preds)
    all_truth = np.concatenate(all_truth)
    return eval_metrics(all_preds, all_truth)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("[EXP-8] 补充对比实验: 多模态融合 SOTA + 领域自适应")
    dap = DefensiveProtocol("EXP8_supplementary")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    d_a, d_m, d_p = data["audio_dim"], data["meta_dim"], data["part_dim"]
    print(f"  N={N}, audio={d_a}, meta={d_m}, part={d_p}")

    results = {"N": N, "baselines": {}}

    # --- 参数量统计 ---
    print("\n  === 模型参数量 ===")
    param_counts = {}
    models_to_count = {
        "SSP-MFN": SSPMFN(d_a, d_m, d_p, 64, 3, N_SCALES, use_gate=True, use_adain=True),
        "MulT": MulT(d_a, d_m, d_p, d_model=64, n_heads=4, n_layers=2, n_out=N_SCALES),
        "TFN": TFN(d_a, d_m, d_p, d_hidden=32, n_out=N_SCALES),
        "LMF": LMF(d_a, d_m, d_p, d_hidden=64, rank=4, n_out=N_SCALES),
        "MLP+MMD": MLPWithMMD(d_a, d_m, d_p, d_hidden=64, n_out=N_SCALES),
        "MLP+CORAL": MLPWithCORAL(d_a, d_m, d_p, d_hidden=64, n_out=N_SCALES),
    }
    for name, m in models_to_count.items():
        pc = count_params(m)
        param_counts[name] = pc
        print(f"    {name:<12}: {pc:>6,} params")
    results["param_counts"] = param_counts

    # --- B9: MulT ---
    print("\n  [B9] Multimodal Transformer...")
    r_mult = run_cv_neural(data, MulT,
        {"d_audio": d_a, "d_meta": d_m, "d_part": d_p,
         "d_model": 64, "n_heads": 4, "n_layers": 2, "n_out": N_SCALES, "dropout": 0.3},
        lr=1e-4)  # MulT 用更小学习率
    results["baselines"]["B9_MulT"] = r_mult
    print(f"    R²={r_mult['_mean']['r2']:.4f}, RMSE={r_mult['_mean']['rmse']:.4f}")

    # --- B10: TFN ---
    print("  [B10] Tensor Fusion Network...")
    r_tfn = run_cv_neural(data, TFN,
        {"d_audio": d_a, "d_meta": d_m, "d_part": d_p,
         "d_hidden": 32, "n_out": N_SCALES, "dropout": 0.3})
    results["baselines"]["B10_TFN"] = r_tfn
    print(f"    R²={r_tfn['_mean']['r2']:.4f}, RMSE={r_tfn['_mean']['rmse']:.4f}")

    # --- B11: LMF ---
    print("  [B11] Low-rank Multimodal Fusion...")
    r_lmf = run_cv_neural(data, LMF,
        {"d_audio": d_a, "d_meta": d_m, "d_part": d_p,
         "d_hidden": 64, "rank": 4, "n_out": N_SCALES, "dropout": 0.3})
    results["baselines"]["B11_LMF"] = r_lmf
    print(f"    R²={r_lmf['_mean']['r2']:.4f}, RMSE={r_lmf['_mean']['rmse']:.4f}")

    # --- B12: MLP+MMD ---
    print("  [B12] MLP + MMD...")
    r_mmd = run_cv_neural(data, MLPWithMMD,
        {"d_audio": d_a, "d_meta": d_m, "d_part": d_p,
         "d_hidden": 64, "n_out": N_SCALES, "dropout": 0.2, "lambda_mmd": 0.1},
        use_domain_loss=True)
    results["baselines"]["B12_MLP_MMD"] = r_mmd
    print(f"    R²={r_mmd['_mean']['r2']:.4f}, RMSE={r_mmd['_mean']['rmse']:.4f}")

    # --- B13: MLP+CORAL ---
    print("  [B13] MLP + CORAL...")
    r_coral = run_cv_neural(data, MLPWithCORAL,
        {"d_audio": d_a, "d_meta": d_m, "d_part": d_p,
         "d_hidden": 64, "n_out": N_SCALES, "dropout": 0.2, "lambda_coral": 0.1},
        use_domain_loss=True)
    results["baselines"]["B13_MLP_CORAL"] = r_coral
    print(f"    R²={r_coral['_mean']['r2']:.4f}, RMSE={r_coral['_mean']['rmse']:.4f}")

    # --- 汇总对比 ---
    print("\n  ============================================================")
    print(f"  {'模型':<15} {'R²':>8} {'RMSE':>8} {'r':>8} {'Params':>8}")
    print(f"  {'-'*50}")
    # 加入 SSP-MFN 参考值 (从 EXP-1 结果)
    try:
        with open(RES/"EXP1_S6_full.json") as f:
            exp1 = json.load(f)
        sspmfn_r2 = exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["r2"]
        sspmfn_rmse = exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["rmse"]
        sspmfn_r = exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["r"]
        print(f"  {'SSP-MFN':<15} {sspmfn_r2:>8.4f} {sspmfn_rmse:>8.4f} {sspmfn_r:>8.4f} {param_counts['SSP-MFN']:>8,}")
        results["SSP_MFN_ref"] = {"r2": sspmfn_r2, "rmse": sspmfn_rmse, "r": sspmfn_r}
    except:
        pass

    for bname, bres in results["baselines"].items():
        short = bname.replace("B9_","").replace("B10_","").replace("B11_","").replace("B12_","").replace("B13_","")
        pc = param_counts.get(short, param_counts.get(short.replace("_"," ").replace("MLP ","MLP+"), "?"))
        print(f"  {short:<15} {bres['_mean']['r2']:>8.4f} {bres['_mean']['rmse']:>8.4f} {bres['_mean']['r']:>8.4f} {pc:>8,}")
    print(f"  ============================================================")

    # DAP
    all_r2 = [r["_mean"]["r2"] for r in results["baselines"].values()]
    dap.check_stability(all_r2)
    results["DAP"] = dap.generate_report()

    out = RES / "EXP8_supplementary_baselines.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
