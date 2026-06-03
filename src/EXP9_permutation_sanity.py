"""EXP-9 · Feature Permutation Sanity Check

验证门控权重的物理意义:
  - 逐模态随机打乱特征
  - 观察 R² 下降量 (ΔR²)
  - 验证 ΔR² 与对应模态的 gate weight 正相关

如果门控权重有物理意义:
  - 打乱 part (gate weight 最高) → R² 下降最多
  - 打乱 audio (gate weight 中等) → R² 下降中等
  - 打乱 meta (gate weight 最低) → R² 下降最少
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"; RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]
N_SCALES = 6

from EXP1_sspmfn_main import build_dataset, MultiModalDS


def train_and_evaluate_fold(data, tr_idx, te_idx, seed=42):
    """训练 SSP-MFN 并返回模型 + 标准化器 + 测试数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr = data["x_audio"][tr_idx]; xa_te = data["x_audio"][te_idx]
    xm_tr = data["x_meta"][tr_idx]; xm_te = data["x_meta"][te_idx]
    xp_tr = data["x_part"][tr_idx]; xp_te = data["x_part"][te_idx]
    y_tr = data["y_adj"][tr_idx]; y_te = data["y_adj"][te_idx]
    eid_tr = data["eth_id"][tr_idx]; eid_te = data["eth_id"][te_idx]

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
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"], d_part=data["part_dim"],
        d_model=64, n_ethnic=3, n_scales=N_SCALES, p_drop=0.3,
        use_adain=True, use_gate=True,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    te_ds = MultiModalDS(xa_te_s, xm_te_s, xp_te_s,
                         sc_y.transform(y_te).astype(np.float32), eid_te)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    best_val, best_state, pat = float("inf"), None, 0
    for ep in range(200):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xa_b, xm_b, xp_b, eid_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                vl.append(criterion(model(xa_b, xm_b, xp_b, eid_b), y_b).item())
        v = np.mean(vl)
        if v < best_val:
            best_val, pat = v, 0
            best_state = {k: v2.cpu().clone() for k, v2 in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 30: break

    model.load_state_dict(best_state)
    model.eval()
    return model, sc_a, sc_m, sc_p, sc_y, xa_te_s, xm_te_s, xp_te_s, y_te, eid_te


def predict_r2(model, xa, xm, xp, eid, y_true, sc_y):
    """用模型预测并计算 mean R²"""
    ds = MultiModalDS(xa, xm, xp, np.zeros_like(y_true, dtype=np.float32),
                      eid.astype(np.int64))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            preds_list.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    preds_s = np.concatenate(preds_list)
    preds = sc_y.inverse_transform(preds_s)
    r2_per_dim = [r2_score(y_true[:, k], preds[:, k]) for k in range(N_SCALES)]
    return np.mean(r2_per_dim), r2_per_dim


def permutation_test(model, xa, xm, xp, eid, y_true, sc_y,
                     modality="audio", n_perm=100, seed=42):
    """对指定模态进行 n_perm 次随机打乱，返回 R² 下降分布"""
    rng = np.random.default_rng(seed)
    r2_orig, r2_orig_per_dim = predict_r2(model, xa, xm, xp, eid, y_true, sc_y)

    delta_r2_list = []
    delta_per_dim = []
    for i in range(n_perm):
        perm_idx = rng.permutation(len(xa))
        if modality == "audio":
            xa_perm = xa[perm_idx]
            r2_perm, r2_perm_dims = predict_r2(model, xa_perm, xm, xp, eid, y_true, sc_y)
        elif modality == "meta":
            xm_perm = xm[perm_idx]
            r2_perm, r2_perm_dims = predict_r2(model, xa, xm_perm, xp, eid, y_true, sc_y)
        elif modality == "part":
            xp_perm = xp[perm_idx]
            r2_perm, r2_perm_dims = predict_r2(model, xa, xm, xp_perm, eid, y_true, sc_y)
        delta_r2_list.append(r2_orig - r2_perm)
        delta_per_dim.append([r2_orig_per_dim[k] - r2_perm_dims[k] for k in range(N_SCALES)])

    return {
        "r2_original": r2_orig,
        "mean_delta_r2": float(np.mean(delta_r2_list)),
        "std_delta_r2": float(np.std(delta_r2_list)),
        "delta_per_dim": np.mean(delta_per_dim, axis=0).tolist(),
    }


def main():
    print("[EXP-9] Feature Permutation Sanity Check")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    # 加载 EXP-5 门控权重作为参考
    with open(RES / "EXP5_gate_weights.json") as f:
        exp5 = json.load(f)
    gate_weights = {
        "audio": exp5["overall_gates"]["audio"]["mean"],
        "meta": exp5["overall_gates"]["meta"]["mean"],
        "part": exp5["overall_gates"]["part"]["mean"],
    }
    print(f"  参考门控权重: audio={gate_weights['audio']:.4f}, "
          f"meta={gate_weights['meta']:.4f}, part={gate_weights['part']:.4f}")

    # 5-fold CV, 每折做 permutation test
    gkf = GroupKFold(n_splits=5)
    all_results = {"audio": [], "meta": [], "part": []}

    for fold, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        print(f"\n  Fold {fold+1}/5: 训练模型...")
        model, sc_a, sc_m, sc_p, sc_y, xa_te, xm_te, xp_te, y_te, eid_te = \
            train_and_evaluate_fold(data, tr, te, seed=42+fold)

        for mod in ["audio", "meta", "part"]:
            print(f"    Permuting {mod} (50 times)...")
            res = permutation_test(model, xa_te, xm_te, xp_te, eid_te, y_te, sc_y,
                                   modality=mod, n_perm=50, seed=fold*100)
            all_results[mod].append(res)

    # 汇总
    print("\n" + "="*70)
    print("  Feature Permutation Results (mean across 5 folds)")
    print("="*70)
    print(f"  {'Modality':<10} {'Gate Weight':>12} {'ΔR² (perm)':>12} {'Rank':>6}")
    print(f"  {'-'*45}")

    summary = {}
    for mod in ["audio", "meta", "part"]:
        mean_delta = np.mean([r["mean_delta_r2"] for r in all_results[mod]])
        std_delta = np.std([r["mean_delta_r2"] for r in all_results[mod]])
        summary[mod] = {
            "gate_weight": gate_weights[mod],
            "mean_delta_r2": float(mean_delta),
            "std_delta_r2": float(std_delta),
            "per_fold": all_results[mod],
        }
        print(f"  {mod:<10} {gate_weights[mod]:>12.4f} {mean_delta:>+12.4f} ± {std_delta:.4f}")

    # 排序一致性检验
    gate_rank = sorted(["audio","meta","part"], key=lambda m: gate_weights[m], reverse=True)
    delta_rank = sorted(["audio","meta","part"],
                        key=lambda m: summary[m]["mean_delta_r2"], reverse=True)
    print(f"\n  门控权重排序: {' > '.join(gate_rank)}")
    print(f"  ΔR² 排序:     {' > '.join(delta_rank)}")
    rank_match = gate_rank == delta_rank
    print(f"  排序一致性:   {'✓ 完全一致' if rank_match else '✗ 不一致'}")

    # Spearman 相关 (逐量表)
    print("\n  === 逐量表 Permutation ΔR² ===")
    print(f"  {'Scale':<8} {'ΔR²_audio':>10} {'ΔR²_meta':>10} {'ΔR²_part':>10}")
    gate_per_scale = exp5.get("per_scale_gates", {})
    for k, s in enumerate(SCALES):
        da = np.mean([r["delta_per_dim"][k] for r in all_results["audio"]])
        dm = np.mean([r["delta_per_dim"][k] for r in all_results["meta"]])
        dp = np.mean([r["delta_per_dim"][k] for r in all_results["part"]])
        print(f"  {s:<8} {da:>+10.4f} {dm:>+10.4f} {dp:>+10.4f}")

    # 计算 gate weight 与 ΔR² 的 Spearman 相关
    gate_vals = []
    delta_vals = []
    for s in SCALES:
        for mod in ["audio", "meta", "part"]:
            gw = gate_per_scale.get(s, {}).get(mod, 0)
            dr = np.mean([r["delta_per_dim"][SCALES.index(s)] for r in all_results[mod]])
            gate_vals.append(gw)
            delta_vals.append(dr)

    rho, p_val = spearmanr(gate_vals, delta_vals)
    r_pearson, p_pearson = pearsonr(gate_vals, delta_vals)
    print(f"\n  Gate Weight vs ΔR² 相关:")
    print(f"    Spearman ρ = {rho:.4f}, p = {p_val:.4f}")
    print(f"    Pearson  r = {r_pearson:.4f}, p = {p_pearson:.4f}")

    summary["correlation"] = {
        "spearman_rho": float(rho), "spearman_p": float(p_val),
        "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
    }
    summary["rank_consistent"] = rank_match

    # DAP
    dap = DefensiveProtocol("EXP9_permutation")
    stab = dap.check_stability([summary[m]["mean_delta_r2"] for m in ["audio","meta","part"]])
    print(f"\n  [DAP] {stab.message}")
    summary["DAP"] = dap.generate_report()

    out = RES / "EXP9_permutation_sanity.json"
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
