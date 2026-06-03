"""EXP-17 · 主管线 (build_data, S6 signal) 上的 AdaIN 证伪性控制实验

对应论文 K1 返修意见: 在主管线 (SIGNAL=1.5, INT_RATIO=0.7, 6-scale R²≈0.119) 上
复现 EXP16 的 C1 (shuffled-label) 与 C2 (zero-embedding) 两组对照, 检验
AdaIN 增益 (ΔR²=+0.059) 是否来自真实 group-conditional 信号而非参数容量。

实验设置:
  C0_full       主管线基线 (真实 eth_id, 完整 AdaIN, 含可训练 embedding)
  C1_shuffled   边缘频率保留下随机置换 eth_id
  C2_zeroembed  冻结 embedding=0, 保留 to_affine 可训练 (参数容量匹配)

防御性协议 (DAP) 集成 5 项检查:
  1. label_leakage   (特征-标签最大 |r|)
  2. stability       (跨种子 CV)
  3. overfit_ratio   (单折 train/test RMSE)
  4. distribution_shift (KS test on 训练/测试特征)
  5. permutation_baseline (随机标签基线)

注: 主管线数据真实读自 数据/数据v2, 但 EXP1_S6_full 中 DATA_DIR 硬编码为 真实数据v2,
脚本启动时显式重定向, 与 EXP16 处理方式保持一致。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

import EXP1_S6_full as s6
s6.DATA_DIR = s6.ROOT / "数据" / "数据v2"
from EXP1_S6_full import (
    build_data, eval_metrics, SCALES, N_SCALES, DEVICE, RES, MMDS,
)
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol


SEEDS = [17, 42, 2024]
SCALES_4 = ["ICS", "IRI", "CSAS", "SSCS"]


def shuffle_eth_id(eth_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eth_id))
    return eth_id[perm]


def patch_zero_embed(model: SSPMFN):
    with torch.no_grad():
        model.adain.embed.weight.data.zero_()
    model.adain.embed.weight.requires_grad_(False)
    return model


def train_with_variant(data, tr, te, seed=42, variant="full"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr, xa_te = data["x_audio"][tr], data["x_audio"][te]
    xm_tr, xm_te = data["x_meta"][tr], data["x_meta"][te]
    xp_tr, xp_te = data["x_part"][tr], data["x_part"][te]
    y_tr, y_te = data["y_adj"][tr], data["y_adj"][te]
    eid_tr, eid_te = data["eth_id"][tr], data["eth_id"][te]

    sa = StandardScaler().fit(xa_tr); xa_tr = sa.transform(xa_tr); xa_te = sa.transform(xa_te)
    sm = StandardScaler().fit(xm_tr); xm_tr = sm.transform(xm_tr); xm_te = sm.transform(xm_te)
    sp = StandardScaler().fit(xp_tr); xp_tr = sp.transform(xp_tr); xp_te = sp.transform(xp_te)
    sy = StandardScaler().fit(y_tr); y_tr_s = sy.transform(y_tr)

    tr_ds = MMDS(xa_tr.astype(np.float32), xm_tr.astype(np.float32),
                 xp_tr.astype(np.float32), y_tr_s.astype(np.float32), eid_tr)
    te_ds = MMDS(xa_te.astype(np.float32), xm_te.astype(np.float32),
                 xp_te.astype(np.float32), sy.transform(y_te).astype(np.float32), eid_te)
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
    crit = nn.HuberLoss(delta=1.0)
    best_val, pat, best_st = float("inf"), 0, None
    for ep in range(200):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xa_b, xm_b, xp_b, eid_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if variant == "zeroembed":
            with torch.no_grad():
                model.adain.embed.weight.data.zero_()
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
            if pat >= 20:
                break

    model.load_state_dict(best_st); model.eval()
    preds_s = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in te_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            eid_b = eid_b.to(DEVICE)
            preds_s.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    return sy.inverse_transform(np.concatenate(preds_s)), y_te


def run_variant_cv(data, variant: str, seed: int = 42, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    ap, at = [], []
    for tr, te in gkf.split(data["x_audio"], data["y_adj"], data["groups"]):
        p, t = train_with_variant(data, tr, te, seed=seed, variant=variant)
        ap.append(p); at.append(t)
    return eval_metrics(np.concatenate(ap), np.concatenate(at))


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
    print(f"[EXP-17] Primary-pipeline falsification · device={DEVICE} · DATA_DIR={s6.DATA_DIR.name}")
    print(f"  S={s6.SIGNAL}, σ={s6.NOISE}, int_ratio={s6.INT_RATIO}")
    dap = DefensiveProtocol("EXP17_falsification_primary")

    data = build_data(seed=42, frac=1.0)
    N = len(data["y_adj"])
    print(f"  N={N}, audio={data['audio_dim']}, meta={data['meta_dim']}, part={data['part_dim']}")

    results = {"n_samples": N, "device": str(DEVICE), "seeds": SEEDS,
               "pipeline": "primary (build_data, S=1.5, int_ratio=0.7)"}

    print("\n[DAP-1] label leakage scan...")
    X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    leak = dap.check_leakage(X_all, data["y_adj"])
    print(f"  {leak.message}")

    print(f"\n[C0_full] true ethnic labels, full AdaIN, {len(SEEDS)} seeds × 5-fold...")
    c0_runs = []
    eth_id_orig = data["eth_id"].copy()
    for s in SEEDS:
        r = run_variant_cv(data, variant="full", seed=s)
        c0_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    results["C0_full"] = summarize(c0_runs)
    results["C0_full"]["per_scale_last_seed"] = {s: c0_runs[-1][s] for s in SCALES}

    print(f"\n[C1_shuffled] permuted ethnic labels (frequency-preserving), {len(SEEDS)} seeds...")
    c1_runs = []
    for s in SEEDS:
        data["eth_id"] = shuffle_eth_id(eth_id_orig, seed=s)
        r = run_variant_cv(data, variant="full", seed=s)
        c1_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    data["eth_id"] = eth_id_orig
    results["C1_shuffled"] = summarize(c1_runs)
    results["C1_shuffled"]["per_scale_last_seed"] = {s: c1_runs[-1][s] for s in SCALES}

    print(f"\n[C2_zeroembed] frozen-zero embedding + trainable affine, {len(SEEDS)} seeds...")
    c2_runs = []
    for s in SEEDS:
        r = run_variant_cv(data, variant="zeroembed", seed=s)
        c2_runs.append(r)
        print(f"  seed={s}: R²_6={r['_mean']['r2']:.4f}")
    results["C2_zeroembed"] = summarize(c2_runs)
    results["C2_zeroembed"]["per_scale_last_seed"] = {s: c2_runs[-1][s] for s in SCALES}

    c0_r2_6 = results["C0_full"]["r2_6_mean"]
    c0_r2_4 = results["C0_full"]["r2_4_mean"]
    delta = {
        "C1_vs_C0_r2_6": round(results["C1_shuffled"]["r2_6_mean"] - c0_r2_6, 4),
        "C1_vs_C0_r2_4": round(results["C1_shuffled"]["r2_4_mean"] - c0_r2_4, 4),
        "C2_vs_C0_r2_6": round(results["C2_zeroembed"]["r2_6_mean"] - c0_r2_6, 4),
        "C2_vs_C0_r2_4": round(results["C2_zeroembed"]["r2_4_mean"] - c0_r2_4, 4),
    }
    results["delta_vs_C0"] = delta

    out_path = RES / "EXP17_falsification_primary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"  [partial save] → {out_path}")

    print("\n[DAP-2] stability across seeds (C0)...")
    stab = dap.check_stability(results["C0_full"]["per_seed_r2_6"], cv_threshold=0.15)
    print(f"  {stab.message}")

    print("\n[DAP-3] overfit on fold-0 (train vs test RMSE)...")
    gkf = GroupKFold(n_splits=5)
    tr, te = next(iter(gkf.split(data["x_audio"], data["y_adj"], data["groups"])))
    preds_te, truth_te = train_with_variant(data, tr, te, variant="full")
    preds_tr, truth_tr = train_with_variant(data, tr, tr, variant="full")
    tr_rmse = float(np.sqrt(mean_squared_error(truth_tr, preds_tr)))
    te_rmse = float(np.sqrt(mean_squared_error(truth_te, preds_te)))
    overfit = dap.check_overfit(tr_rmse, te_rmse)
    print(f"  {overfit.message}")

    print("\n[DAP-4] train/test distribution shift (fold-0)...")
    shift = dap.check_distribution_shift(X_all[tr], X_all[te])
    print(f"  {shift.message}")

    print("\n[DAP-5] permutation baseline (random labels)...")
    model_rmse = results["C0_full"]["rmse_6_mean"]
    perm = dap.check_permutation_baseline(model_rmse, data["y_adj"][:, 0])
    print(f"  {perm.message}")

    dap_report = dap.generate_report()
    results["DAP"] = dap_report
    results["elapsed_sec"] = round(time.time() - t0, 1)

    out_path = RES / "EXP17_falsification_primary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\nsaved → {out_path}")

    print("\n" + "=" * 78)
    print("  EXP-17 Primary-Pipeline Falsification Controls Summary")
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
    print(f"\n  C1 − C0 (shuffled vs true): R²_6 Δ={delta['C1_vs_C0_r2_6']:+.4f}, R²_4 Δ={delta['C1_vs_C0_r2_4']:+.4f}")
    print(f"  C2 − C0 (zero-emb vs true): R²_6 Δ={delta['C2_vs_C0_r2_6']:+.4f}, R²_4 Δ={delta['C2_vs_C0_r2_4']:+.4f}")
    print(f"  elapsed: {results['elapsed_sec']}s")
    print("\nInterpretation rule:")
    print("  • C1 显著低于 C0 AND C2 显著低于 C0  → AdaIN 增益归因于 group-conditional 信号")
    print("  • C1 ≈ C0 OR C2 ≈ C0                  → 增益主要来自参数容量")


if __name__ == "__main__":
    main()
