"""EXP-16 · AdaIN 增益的证伪性控制实验

回应审稿意见 R-A1: AdaIN 在主管线中产生 ΔR²=+0.059 的增益,
但该增益可能源自 (a) 真实的 group-conditional 结构, 或 (b) 额外参数带来的容量增益。
两个证伪实验:

  C0_full       基线: 真实民族标签 + 完整 AdaIN (重跑作为参考)
  C1_shuffled   置换标签控制: 在保留民族边缘频率的前提下随机置换 eth_id
                若 AdaIN 增益来自 group-conditional 信号, C1 应显著低于 C0
  C2_zeroembed  容量匹配控制: 冻结 embedding 为零, 保留可训练 affine 层
                若 AdaIN 增益来自参数容量, C2 应接近 C0

并集成 5 项 DAP 检查 (leakage / stability / overfit / shift / permutation).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
import EXP1_sspmfn_main as exp1
# 修正数据路径: 实际目录为 数据v2 而非 真实数据v2
exp1.DATA_DIR = exp1.ROOT / "数据" / "数据v2"
from EXP1_sspmfn_main import (
    build_dataset, train_sspmfn, train_baseline_ridge, eval_metrics,
    run_cv, SCALES, N_SCALES, DEVICE, RES,
)
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol


SEEDS = [17, 42, 2024]
SCALES_4 = ["ICS", "IRI", "CSAS", "SSCS"]


def shuffle_eth_id(eth_id: np.ndarray, seed: int) -> np.ndarray:
    """边缘频率保持下的随机置换"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eth_id))
    return eth_id[perm]


def patch_zero_embed(model: SSPMFN):
    """将 AdaIN 的 ethnic embedding 冻结为零, 保留 to_affine 可训练
    capacity-matched control: 输入恒为零, gamma/beta 退化为可训练偏置
    """
    with torch.no_grad():
        model.adain.embed.weight.data.zero_()
    model.adain.embed.weight.requires_grad_(False)
    return model


def train_with_variant(data, tr_idx, te_idx, seed=42, variant="full"):
    """复刻 train_sspmfn 但允许在模型构造后注入 zero_embed 补丁"""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    from EXP1_sspmfn_main import MultiModalDS

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
    sc_y = StandardScaler().fit(y_tr); y_tr_s = sc_y.transform(y_tr); y_te_s = sc_y.transform(y_te)

    tr_ds = MultiModalDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                         xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MultiModalDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                         xp_te.astype(np.float32), y_te_s.astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=32, shuffle=False)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"], d_part=data["part_dim"],
        d_model=64, n_ethnic=3, n_scales=6, p_drop=0.3,
        use_adain=True, use_gate=True,
    ).to(DEVICE)

    if variant == "zeroembed":
        model = patch_zero_embed(model)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    epochs = 200
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.HuberLoss(delta=1.0)
    best_loss = float("inf"); best_state = None; patience_cnt = 0; patience = 30

    for ep in range(epochs):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            pred = model(xa_b, xm_b, xp_b, eid_b)
            loss = crit(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # zero-embed 变体确保权重始终为零 (虽然 requires_grad=False, 双保险)
        if variant == "zeroembed":
            with torch.no_grad():
                model.adain.embed.weight.data.zero_()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                pred = model(xa_b, xm_b, xp_b, eid_b)
                val_losses.append(crit(pred, y_b).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss; patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds_s = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, y_b, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            out = model(xa_b, xm_b, xp_b, eid_b, return_weights=False)
            preds_s.append(out.cpu().numpy())
    preds_s = np.concatenate(preds_s)
    preds = sc_y.inverse_transform(preds_s)
    return preds, y_te, None, None, ep + 1


def run_variant_cv(data, variant: str, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    all_preds, all_truth = [], []
    for tr, te in gkf.split(data["x_audio"], data["y_adj"], data["groups"]):
        preds, truth, _, _, _ = train_with_variant(data, tr, te, variant=variant)
        all_preds.append(preds); all_truth.append(truth)
    all_preds = np.concatenate(all_preds)
    all_truth = np.concatenate(all_truth)
    return eval_metrics(all_preds, all_truth)


def summarize(metric_dict_list):
    r2_6 = [m["_mean"]["r2"] for m in metric_dict_list]
    r2_4 = [float(np.mean([m[s]["r2"] for s in SCALES_4])) for m in metric_dict_list]
    rmse_6 = [m["_mean"]["rmse"] for m in metric_dict_list]
    return {
        "r2_6_mean": round(float(np.mean(r2_6)), 4),
        "r2_6_std": round(float(np.std(r2_6)), 4),
        "r2_4_mean": round(float(np.mean(r2_4)), 4),
        "r2_4_std": round(float(np.std(r2_4)), 4),
        "rmse_6_mean": round(float(np.mean(rmse_6)), 4),
        "per_seed_r2_6": [round(x, 4) for x in r2_6],
        "per_seed_r2_4": [round(x, 4) for x in r2_4],
    }


def main():
    t0 = time.time()
    print(f"[EXP-16] Falsification controls · device={DEVICE}")
    dap = DefensiveProtocol("EXP16_falsification")

    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}, audio={data['audio_dim']}, meta={data['meta_dim']}, part={data['part_dim']}")

    results = {"n_samples": N, "device": str(DEVICE), "seeds": SEEDS}

    # DAP 1: leakage on full dataset
    print("\n[DAP-1] label leakage scan...")
    X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    leak = dap.check_leakage(X_all, data["y_adj"])
    print(f"  {leak.message}")

    # === C0: true labels (reference) ===
    print(f"\n[C0_full] true ethnic labels, full AdaIN, {len(SEEDS)} seeds × 5-fold...")
    c0_runs = []
    eth_id_orig = data["eth_id"].copy()
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        r = run_variant_cv(data, variant="full")
        c0_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    results["C0_full"] = summarize(c0_runs)
    results["C0_full"]["per_scale_last_seed"] = {s: c0_runs[-1][s] for s in SCALES}

    # === C1: shuffled labels ===
    print(f"\n[C1_shuffled] permuted ethnic labels (frequency-preserving), {len(SEEDS)} seeds...")
    c1_runs = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        data["eth_id"] = shuffle_eth_id(eth_id_orig, seed=s)
        r = run_variant_cv(data, variant="full")
        c1_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    data["eth_id"] = eth_id_orig
    results["C1_shuffled"] = summarize(c1_runs)
    results["C1_shuffled"]["per_scale_last_seed"] = {s: c1_runs[-1][s] for s in SCALES}

    # === C2: zero embedding ===
    print(f"\n[C2_zeroembed] frozen-zero embedding + trainable affine, {len(SEEDS)} seeds...")
    c2_runs = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        r = run_variant_cv(data, variant="zeroembed")
        c2_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    results["C2_zeroembed"] = summarize(c2_runs)
    results["C2_zeroembed"]["per_scale_last_seed"] = {s: c2_runs[-1][s] for s in SCALES}

    # === 差异表 (相对 C0) ===
    c0_r2_6 = results["C0_full"]["r2_6_mean"]
    c0_r2_4 = results["C0_full"]["r2_4_mean"]
    delta = {
        "C1_vs_C0_r2_6": round(results["C1_shuffled"]["r2_6_mean"] - c0_r2_6, 4),
        "C1_vs_C0_r2_4": round(results["C1_shuffled"]["r2_4_mean"] - c0_r2_4, 4),
        "C2_vs_C0_r2_6": round(results["C2_zeroembed"]["r2_6_mean"] - c0_r2_6, 4),
        "C2_vs_C0_r2_4": round(results["C2_zeroembed"]["r2_4_mean"] - c0_r2_4, 4),
    }
    results["delta_vs_C0"] = delta

    # 提前保存核心结果, 避免 DAP 阶段崩溃丢数
    out_path = RES / "EXP16_falsification_controls.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"  [partial save] → {out_path}")

    # === DAP 2: stability across seeds (C0) ===
    print("\n[DAP-2] stability across seeds (C0)...")
    stab = dap.check_stability(results["C0_full"]["per_seed_r2_6"], cv_threshold=0.15)
    print(f"  {stab.message}")

    # === DAP 3: overfit on a single fold ===
    print("\n[DAP-3] overfit on fold-0 (train vs test RMSE)...")
    gkf = GroupKFold(n_splits=5)
    tr, te = next(iter(gkf.split(data["x_audio"], data["y_adj"], data["groups"])))
    preds_te, truth_te, _, _, _ = train_with_variant(data, tr, te, variant="full")
    preds_tr, truth_tr, _, _, _ = train_with_variant(data, tr, tr, variant="full")
    tr_rmse = float(np.sqrt(mean_squared_error(truth_tr, preds_tr)))
    te_rmse = float(np.sqrt(mean_squared_error(truth_te, preds_te)))
    overfit = dap.check_overfit(tr_rmse, te_rmse)
    print(f"  {overfit.message}")

    # === DAP 4: distribution shift on fold-0 ===
    print("\n[DAP-4] train/test distribution shift (fold-0)...")
    shift = dap.check_distribution_shift(X_all[tr], X_all[te])
    print(f"  {shift.message}")

    # === DAP 5: permutation baseline ===
    print("\n[DAP-5] permutation baseline (random labels)...")
    model_rmse = results["C0_full"]["rmse_6_mean"]
    perm = dap.check_permutation_baseline(model_rmse, data["y_adj"][:, 0])
    print(f"  {perm.message}")

    # === 汇总 + 报告 ===
    dap_report = dap.generate_report()
    results["DAP"] = dap_report
    results["elapsed_sec"] = round(time.time() - t0, 1)

    out_path = RES / "EXP16_falsification_controls.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\nsaved → {out_path}")

    # 终端汇总
    print("\n" + "=" * 78)
    print("  EXP-16 Falsification Controls Summary")
    print("=" * 78)
    hdr = f"  {'Variant':<22} {'R²_6 (mean±std)':<22} {'R²_4 (mean±std)':<22} {'ΔR²_6':<10}"
    print(hdr)
    print("  " + "-" * 75)
    for name, key in [("C0_full (reference)", "C0_full"),
                      ("C1_shuffled labels", "C1_shuffled"),
                      ("C2_zero-embedding", "C2_zeroembed")]:
        v = results[key]
        d6 = v["r2_6_mean"] - c0_r2_6
        line = f"  {name:<22} {v['r2_6_mean']:.4f}±{v['r2_6_std']:.4f}     {v['r2_4_mean']:.4f}±{v['r2_4_std']:.4f}     {d6:+.4f}"
        print(line)
    print("=" * 78)
    print(f"\n  C1 - C0 (shuffled vs true): R²_6 Δ={delta['C1_vs_C0_r2_6']:+.4f}, R²_4 Δ={delta['C1_vs_C0_r2_4']:+.4f}")
    print(f"  C2 - C0 (zero-emb vs true): R²_6 Δ={delta['C2_vs_C0_r2_6']:+.4f}, R²_4 Δ={delta['C2_vs_C0_r2_4']:+.4f}")
    print(f"  elapsed: {results['elapsed_sec']}s")
    print("\nInterpretation:")
    print("  • |C1 - C0| large AND |C2 - C0| large  → AdaIN gain attributable to group structure")
    print("  • |C1 - C0| ≈ 0   OR  |C2 - C0| ≈ 0    → gain likely from capacity not group identity")


if __name__ == "__main__":
    main()
