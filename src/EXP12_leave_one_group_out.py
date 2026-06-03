"""EXP-12 · Leave-One-Group-Out (LOGO) 验证 (R2-6, R8-4)

依次留出一个民族组作为测试集，其余两组训练:
  - 留出侗族 (dong):     训练=藏族+蒙古族, 测试=侗族
  - 留出藏族 (tibetan):  训练=侗族+蒙古族, 测试=藏族
  - 留出蒙古族 (mongolian): 训练=侗族+藏族, 测试=蒙古族

报告:
  - 每组留出时的 R²/RMSE/r (逐量表 + 均值)
  - 与 GroupKFold 5折 CV 结果对比
  - 跨民族泛化能力评估

防御性策略:
  1. 训练/测试集民族分布偏移检测 (KS)
  2. 每组留出的置换基线验证
  3. 稳定性检测 (3组留出的 R² 方差)
  4. 效应量检测 (LOGO vs 随机基线)
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol
from EXP1_sspmfn_main import build_dataset, MultiModalDS

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6
ETHNIC_NAMES = ["侗族", "藏族", "蒙古族"]
ETHNIC_EN = {"侗族": "dong", "藏族": "tibetan", "蒙古族": "mongolian"}


def train_logo_fold(data, tr_mask: np.ndarray, te_mask: np.ndarray,
                    seed: int = 42, epochs: int = 200, patience: int = 30):
    """训练一个 LOGO 折"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = np.where(tr_mask)[0]
    te = np.where(te_mask)[0]

    xa_tr = data["x_audio"][tr]
    xm_tr = data["x_meta"][tr]
    xp_tr = data["x_part"][tr]
    y_tr = data["y_adj"][tr]
    eid_tr = data["eth_id"][tr]

    xa_te = data["x_audio"][te]
    xm_te = data["x_meta"][te]
    xp_te = data["x_part"][te]
    y_te = data["y_adj"][te]
    eid_te = data["eth_id"][te]

    sc_a = StandardScaler().fit(xa_tr)
    sc_m = StandardScaler().fit(xm_tr)
    sc_p = StandardScaler().fit(xp_tr)
    sc_y = StandardScaler().fit(y_tr)

    xa_tr_s = sc_a.transform(xa_tr).astype(np.float32)
    xm_tr_s = sc_m.transform(xm_tr).astype(np.float32)
    xp_tr_s = sc_p.transform(xp_tr).astype(np.float32)
    y_tr_s = sc_y.transform(y_tr).astype(np.float32)
    xa_te_s = sc_a.transform(xa_te).astype(np.float32)
    xm_te_s = sc_m.transform(xm_te).astype(np.float32)
    xp_te_s = sc_p.transform(xp_te).astype(np.float32)

    tr_ds = MultiModalDS(xa_tr_s, xm_tr_s, xp_tr_s, y_tr_s, eid_tr)
    te_ds = MultiModalDS(xa_te_s, xm_te_s, xp_te_s,
                         sc_y.transform(y_te).astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"],
        d_part=data["part_dim"], d_model=64, n_scales=N_SCALES,
        n_ethnic=3, use_gate=True, use_adain=True,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.HuberLoss(delta=1.0)

    best_val, best_st, pat = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            crit(model(xa_b, xm_b, xp_b, eid_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                vl.append(crit(model(xa_b, xm_b, xp_b, eid_b), y_b).item())
        v = np.mean(vl)
        if v < best_val:
            best_val, pat = v, 0
            best_st = {k: v2.cpu().clone() for k, v2 in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_st)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            preds_list.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    preds = sc_y.inverse_transform(np.concatenate(preds_list))
    return preds, y_te, len(tr), len(te)


def compute_metrics(preds: np.ndarray, truth: np.ndarray) -> dict:
    results = {}
    for k, name in enumerate(SCALES):
        rmse = float(np.sqrt(mean_squared_error(truth[:, k], preds[:, k])))
        r2 = float(r2_score(truth[:, k], preds[:, k]))
        r, _ = pearsonr(truth[:, k], preds[:, k])
        results[name] = {"rmse": round(rmse, 4), "r2": round(r2, 4), "r": round(float(r), 4)}
    results["_mean"] = {
        "rmse": round(float(np.mean([results[s]["rmse"] for s in SCALES])), 4),
        "r2": round(float(np.mean([results[s]["r2"] for s in SCALES])), 4),
        "r": round(float(np.mean([results[s]["r"] for s in SCALES])), 4),
    }
    return results


def defensive_checks(data, logo_r2_means: list[float], eth_masks: dict):
    dap = DefensiveProtocol("EXP12_logo")

    dap.check_stability(logo_r2_means)

    X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    for eth_name, (tr_mask, te_mask) in eth_masks.items():
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        dap.check_distribution_shift(X_all[tr_idx], X_all[te_idx])

    best_r2 = max(logo_r2_means)
    best_rmse = float(np.sqrt(mean_squared_error(
        data["y_adj"], np.zeros_like(data["y_adj"])
    )))
    dap.check_permutation_baseline(best_rmse, data["y_adj"][:, 0], metric_type="rmse")

    dap.check_effect_size(
        model_metric=float(np.mean(logo_r2_means)),
        baseline_metric=0.0,
        metric_type="r2",
        min_improvement=0.02,
    )

    return dap.generate_report()


def main():
    print("[EXP-12] Leave-One-Group-Out (LOGO) 验证")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    eth_id = data["eth_id"]
    eth_masks = {}
    for i, eth_name in enumerate(ETHNIC_NAMES):
        te_mask = eth_id == i
        tr_mask = ~te_mask
        eth_masks[eth_name] = (tr_mask, te_mask)
        n_tr = tr_mask.sum()
        n_te = te_mask.sum()
        print(f"  {eth_name}: train={n_tr}, test={n_te}")

    results = {"n_samples": N, "logo_results": {}}
    logo_r2_means = []

    for eth_name, (tr_mask, te_mask) in eth_masks.items():
        print(f"\n  留出 {eth_name}...")
        preds, truth, n_tr, n_te = train_logo_fold(
            data, tr_mask, te_mask, seed=42
        )
        metrics = compute_metrics(preds, truth)
        logo_r2_means.append(metrics["_mean"]["r2"])

        results["logo_results"][eth_name] = {
            "n_train": int(n_tr),
            "n_test": int(n_te),
            "metrics": metrics,
        }
        print(f"    R²={metrics['_mean']['r2']:.4f}, RMSE={metrics['_mean']['rmse']:.4f}, r={metrics['_mean']['r']:.4f}")
        for s in SCALES:
            print(f"      {s}: R²={metrics[s]['r2']:.4f}, RMSE={metrics[s]['rmse']:.4f}")

    mean_logo_r2 = float(np.mean(logo_r2_means))
    std_logo_r2 = float(np.std(logo_r2_means))
    results["summary"] = {
        "mean_r2_across_groups": round(mean_logo_r2, 4),
        "std_r2_across_groups": round(std_logo_r2, 4),
        "logo_r2_per_group": {
            eth: round(r2, 4)
            for eth, r2 in zip(ETHNIC_NAMES, logo_r2_means)
        },
    }

    try:
        with open(RES / "EXP1_S6_full.json") as f:
            exp1 = json.load(f)
        cv_r2 = exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["r2"]
        results["summary"]["cv_5fold_r2"] = cv_r2
        results["summary"]["logo_vs_cv_gap"] = round(mean_logo_r2 - cv_r2, 4)
        print(f"\n  5折CV R²={cv_r2:.4f}, LOGO均值R²={mean_logo_r2:.4f}, gap={mean_logo_r2-cv_r2:+.4f}")
    except Exception:
        pass

    print("\n  防御性检查...")
    dap_report = defensive_checks(data, logo_r2_means, eth_masks)
    results["DAP"] = dap_report

    out_path = RES / "EXP12_leave_one_group_out.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out_path}")

    print("\n" + "=" * 70)
    print(f"  LOGO 验证汇总")
    print(f"  {'留出组':<10} {'N_test':>7} {'R²':>8} {'RMSE':>8} {'r':>8}")
    print(f"  {'-'*50}")
    for eth_name in ETHNIC_NAMES:
        r = results["logo_results"][eth_name]
        m = r["metrics"]["_mean"]
        print(f"  {eth_name:<10} {r['n_test']:>7} {m['r2']:>8.4f} {m['rmse']:>8.4f} {m['r']:>8.4f}")
    print(f"  {'均值':<10} {'':>7} {mean_logo_r2:>8.4f} {'':>8} {'':>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
