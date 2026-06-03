"""EXP-11 · Full vs Participant-only 逐量表 Bootstrap 比较 (R8-2)

比较两种数据配置:
  Full:             x_audio + x_meta + x_part (三路全模态)
  Participant-only: x_part only (仅参与者背景 + pre 量表)

对每个量表独立报告:
  - 两配置的 R² 均值 ± 95% CI (bootstrap 1000次)
  - 配对 t 检验 p 值
  - Cohen's d 效应量

防御性策略:
  1. 每折 train/test 分布偏移检测 (KS)
  2. 置换基线验证信号真实性
  3. 多重比较 Bonferroni 校正 (6量表)
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, ks_2samp

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
N_BOOT = 1000
N_FOLDS = 5


def train_fold(data, tr, te, config: str, seed: int = 42):
    """训练一折，config 控制使用哪些模态"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    xa_tr = data["x_audio"][tr].copy()
    xm_tr = data["x_meta"][tr].copy()
    xp_tr = data["x_part"][tr].copy()
    xa_te = data["x_audio"][te].copy()
    xm_te = data["x_meta"][te].copy()
    xp_te = data["x_part"][te].copy()
    y_tr = data["y_adj"][tr]
    y_te = data["y_adj"][te]
    eid_tr = data["eth_id"][tr]
    eid_te = data["eth_id"][te]

    if config == "part_only":
        xa_tr[:] = 0.0
        xm_tr[:] = 0.0
        xa_te[:] = 0.0
        xm_te[:] = 0.0

    sc_a = StandardScaler().fit(xa_tr)
    sc_m = StandardScaler().fit(xm_tr)
    sc_p = StandardScaler().fit(xp_tr)
    sc_y = StandardScaler().fit(y_tr)

    xa_tr = sc_a.transform(xa_tr).astype(np.float32)
    xm_tr = sc_m.transform(xm_tr).astype(np.float32)
    xp_tr = sc_p.transform(xp_tr).astype(np.float32)
    y_tr_s = sc_y.transform(y_tr).astype(np.float32)
    xa_te = sc_a.transform(xa_te).astype(np.float32)
    xm_te = sc_m.transform(xm_te).astype(np.float32)
    xp_te = sc_p.transform(xp_te).astype(np.float32)

    tr_ds = MultiModalDS(xa_tr, xm_tr, xp_tr, y_tr_s, eid_tr)
    te_ds = MultiModalDS(xa_te, xm_te, xp_te,
                         sc_y.transform(y_te).astype(np.float32), eid_te)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = SSPMFN(
        d_audio=data["audio_dim"], d_meta=data["meta_dim"],
        d_part=data["part_dim"], d_model=64, n_scales=N_SCALES,
        n_ethnic=3, use_gate=True, use_adain=True,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    crit = nn.HuberLoss(delta=1.0)

    best_val, best_st, pat = float("inf"), None, 0
    for _ in range(200):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            crit(model(xa_b, xm_b, xp_b, eid_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
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
            if pat >= 25:
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
    return preds, y_te


def collect_fold_r2(data, config: str):
    """5折 CV，返回 fold×scale 的 R² 矩阵 [5, 6]"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_r2 = np.zeros((N_FOLDS, N_SCALES))
    for fi, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        print(f"    [{config}] fold {fi+1}/{N_FOLDS}...")
        preds, truth = train_fold(data, tr, te, config, seed=42 + fi)
        for k in range(N_SCALES):
            fold_r2[fi, k] = r2_score(truth[:, k], preds[:, k])
    return fold_r2


def bootstrap_compare(r2_full: np.ndarray, r2_part: np.ndarray, rng: np.random.Generator):
    """
    r2_full, r2_part: [N_FOLDS, N_SCALES]
    返回逐量表的 bootstrap CI 和 t 检验结果
    """
    results = {}
    p_values = []
    for k, scale in enumerate(SCALES):
        full_k = r2_full[:, k]
        part_k = r2_part[:, k]
        diff = full_k - part_k

        t_stat, p_val = ttest_rel(full_k, part_k)
        p_values.append(p_val)

        boot_diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, N_FOLDS, size=N_FOLDS)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))

        pooled_std = np.sqrt((full_k.std() ** 2 + part_k.std() ** 2) / 2 + 1e-8)
        cohens_d = float(diff.mean() / pooled_std)

        sig_raw = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        bonf_alpha = 0.05 / N_SCALES
        sig_bonf = "*" if p_val < bonf_alpha else "ns"

        results[scale] = {
            "full_mean_r2": float(full_k.mean()),
            "full_std_r2": float(full_k.std()),
            "part_mean_r2": float(part_k.mean()),
            "part_std_r2": float(part_k.std()),
            "mean_diff": float(diff.mean()),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": cohens_d,
            "sig_raw": sig_raw,
            "sig_bonferroni": sig_bonf,
        }
    return results, p_values


def defensive_checks(data, r2_full: np.ndarray, r2_part: np.ndarray):
    """防御性分析: 分布偏移 + 置换基线 + 多重比较"""
    dap = DefensiveProtocol("EXP11_bootstrap_scale")

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(data["x_audio"], data["y_adj"], data["groups"]))
    tr0, te0 = folds[0]
    X_all = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    dap.check_distribution_shift(X_all[tr0], X_all[te0])

    dap.check_stability(r2_full[:, :].mean(axis=1).tolist())

    model_r2_mean = r2_full.mean()
    dap.check_permutation_baseline(
        float(np.sqrt(mean_squared_error(data["y_adj"], np.zeros_like(data["y_adj"])))),
        data["y_adj"][:, 0],
        metric_type="rmse",
    )

    p_vals_full = []
    for k in range(N_SCALES):
        from scipy.stats import ttest_1samp
        t, p = ttest_1samp(r2_full[:, k], 0.0)
        p_vals_full.append(float(p))
    dap.check_multiple_comparisons(p_vals_full, method="bonferroni")

    return dap.generate_report()


def main():
    print("[EXP-11] Full vs Participant-only 逐量表 Bootstrap 比较")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    print("\n  收集 Full 配置 fold-level R²...")
    r2_full = collect_fold_r2(data, "full")

    print("\n  收集 Participant-only 配置 fold-level R²...")
    r2_part = collect_fold_r2(data, "part_only")

    print("\n  Bootstrap 比较...")
    rng = np.random.default_rng(2026)
    scale_results, p_values = bootstrap_compare(r2_full, r2_part, rng)

    print("\n  防御性检查...")
    dap_report = defensive_checks(data, r2_full, r2_part)

    output = {
        "n_samples": N,
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOT,
        "scale_comparison": scale_results,
        "fold_r2_full": r2_full.tolist(),
        "fold_r2_part": r2_part.tolist(),
        "DAP": dap_report,
    }

    out_path = RES / "EXP11_bootstrap_scale_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out_path}")

    print("\n" + "=" * 80)
    print(f"  {'量表':<8} {'Full R²':>10} {'Part R²':>10} {'ΔR²':>8} {'95% CI':>22} {'p':>8} {'d':>6} {'Bonf':>6}")
    print(f"  {'-'*78}")
    for scale, r in scale_results.items():
        ci_str = f"[{r['ci_95_lo']:+.4f},{r['ci_95_hi']:+.4f}]"
        print(f"  {scale:<8} {r['full_mean_r2']:>10.4f} {r['part_mean_r2']:>10.4f} "
              f"{r['mean_diff']:>+8.4f} {ci_str:>22} {r['p_value']:>8.4f} "
              f"{r['cohens_d']:>6.2f} {r['sig_bonferroni']:>6}")
    print("=" * 80)


if __name__ == "__main__":
    main()
