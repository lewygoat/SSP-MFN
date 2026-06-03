"""EXP-18 · LOGO 协议下的 M3 (No-AdaIN) 与 M4 (Plain) 消融

对应论文 K3 返修意见: 检验 LOGO 负 R² 是否专属于 AdaIN 依赖架构,
还是 SSP-MFN 任一变体在跨民族迁移上都会失败。

变体设置 (与主表一致):
  M1_full       use_adain=True,  use_gate=True   (Table S3 已报告)
  M3_no_adain   use_adain=False, use_gate=True
  M4_plain      use_adain=False, use_gate=False

防御性协议 (DAP) 覆盖:
  1. 分布偏移 (KS test, 训练 vs 测试)
  2. 稳定性 (跨留出组 R² CV)
  3. 置换基线
  4. 效应量 vs 零基线
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
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
import EXP1_sspmfn_main as e1
e1.DATA_DIR = e1.ROOT / "数据" / "数据v2"
from EXP1_sspmfn_main import build_dataset, MultiModalDS


ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
ETHNIC_NAMES = ["侗族", "藏族", "蒙古族"]

VARIANTS = {
    "M3_no_adain": dict(use_adain=False, use_gate=True),
    "M4_plain":    dict(use_adain=False, use_gate=False),
}


def train_logo_fold(data, tr_mask, te_mask, variant_cfg, seed=42,
                    epochs=200, patience=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = np.where(tr_mask)[0]
    te = np.where(te_mask)[0]

    xa_tr, xm_tr, xp_tr = data["x_audio"][tr], data["x_meta"][tr], data["x_part"][tr]
    xa_te, xm_te, xp_te = data["x_audio"][te], data["x_meta"][te], data["x_part"][te]
    y_tr, y_te = data["y_adj"][tr], data["y_adj"][te]
    eid_tr, eid_te = data["eth_id"][tr], data["eth_id"][te]

    sa = StandardScaler().fit(xa_tr); xa_tr = sa.transform(xa_tr); xa_te = sa.transform(xa_te)
    sm = StandardScaler().fit(xm_tr); xm_tr = sm.transform(xm_tr); xm_te = sm.transform(xm_te)
    sp = StandardScaler().fit(xp_tr); xp_tr = sp.transform(xp_tr); xp_te = sp.transform(xp_te)
    sy = StandardScaler().fit(y_tr); y_tr_s = sy.transform(y_tr)

    tr_ds = MultiModalDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                         xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MultiModalDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                         xp_te.astype(np.float32),
                         sy.transform(y_te).astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"],
        d_part=data["part_dim"], d_model=64, n_scales=len(SCALES),
        n_ethnic=3, **variant_cfg,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
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
        sched.step()
        model.eval(); vl = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                vl.append(crit(model(xa_b, xm_b, xp_b, eid_b), y_b).item())
        v = float(np.mean(vl))
        if v < best_val:
            best_val, pat = v, 0
            best_st = {k: v2.cpu().clone() for k, v2 in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_st); model.eval()
    preds_list = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            preds_list.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    return sy.inverse_transform(np.concatenate(preds_list)), y_te


def compute_metrics(preds, truth):
    out = {}
    for k, s in enumerate(SCALES):
        rmse = float(np.sqrt(mean_squared_error(truth[:, k], preds[:, k])))
        r2 = float(r2_score(truth[:, k], preds[:, k]))
        r, _ = pearsonr(truth[:, k], preds[:, k])
        out[s] = {"rmse": round(rmse, 4), "r2": round(r2, 4), "r": round(float(r), 4)}
    out["_mean"] = {
        "rmse": round(float(np.mean([out[s]["rmse"] for s in SCALES])), 4),
        "r2": round(float(np.mean([out[s]["r2"] for s in SCALES])), 4),
        "r": round(float(np.mean([out[s]["r"] for s in SCALES])), 4),
    }
    return out


def main():
    print("[EXP-18] LOGO ablation · M3 (no AdaIN) · M4 (plain)")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    eid = data["eth_id"]
    eth_masks = {n: (eid != i, eid == i) for i, n in enumerate(ETHNIC_NAMES)}
    print(f"  N={N}, device={DEVICE}")

    results = {"n_samples": N, "device": str(DEVICE), "variants": {}}

    for vname, vcfg in VARIANTS.items():
        print(f"\n[{vname}] config={vcfg}")
        dap = DefensiveProtocol(f"EXP18_{vname}")
        per_group = {}
        r2_list = []
        for eth_name, (tr_mask, te_mask) in eth_masks.items():
            print(f"  hold-out {eth_name}...", flush=True)
            preds, truth = train_logo_fold(data, tr_mask, te_mask, vcfg, seed=42)
            m = compute_metrics(preds, truth)
            r2_list.append(m["_mean"]["r2"])
            per_group[eth_name] = {
                "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                "metrics": m,
            }
            print(f"    R²={m['_mean']['r2']:.4f}, RMSE={m['_mean']['rmse']:.4f}, r={m['_mean']['r']:.4f}")

        dap.check_stability(r2_list, cv_threshold=0.5)
        X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
        for eth_name, (tr_mask, te_mask) in eth_masks.items():
            dap.check_distribution_shift(X_all[np.where(tr_mask)[0]],
                                         X_all[np.where(te_mask)[0]])
        rmse_zero = float(np.sqrt(mean_squared_error(
            data["y_adj"], np.zeros_like(data["y_adj"]))))
        dap.check_permutation_baseline(rmse_zero, data["y_adj"][:, 0], metric_type="rmse")
        dap.check_effect_size(model_metric=float(np.mean(r2_list)),
                              baseline_metric=0.0, metric_type="r2", min_improvement=0.02)

        results["variants"][vname] = {
            "config": vcfg,
            "per_group": per_group,
            "summary": {
                "mean_r2": round(float(np.mean(r2_list)), 4),
                "std_r2": round(float(np.std(r2_list)), 4),
                "per_group_r2": {n: round(r, 4) for n, r in zip(ETHNIC_NAMES, r2_list)},
            },
            "DAP": dap.generate_report(),
        }

    try:
        with open(RES / "EXP12_leave_one_group_out.json") as f:
            m1 = json.load(f)
        results["variants"]["M1_full_reference"] = m1.get("summary", {})
    except Exception:
        pass

    out_path = RES / "EXP18_logo_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nsaved → {out_path}")

    print("\n" + "=" * 70)
    print(f"  EXP-18 LOGO ablation summary")
    print(f"  {'Variant':<14} {'Dong':>8} {'Tibetan':>8} {'Mongolian':>10} {'mean R²':>10}")
    print("  " + "-" * 55)
    for vname in VARIANTS:
        s = results["variants"][vname]["summary"]
        pg = s["per_group_r2"]
        print(f"  {vname:<14} {pg['侗族']:>8.4f} {pg['藏族']:>8.4f} {pg['蒙古族']:>10.4f} {s['mean_r2']:>10.4f}")
    if "M1_full_reference" in results["variants"]:
        r = results["variants"]["M1_full_reference"]
        if "logo_r2_per_group" in r:
            pg = r["logo_r2_per_group"]
            print(f"  {'M1_full (ref)':<14} {pg['侗族']:>8.4f} {pg['藏族']:>8.4f} {pg['蒙古族']:>10.4f} {r['mean_r2_across_groups']:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
