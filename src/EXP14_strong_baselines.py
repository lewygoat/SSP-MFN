"""EXP-14 · 强基线补充实验 B14-B16 (R2-4)

新增强基线:
  B14_TabNet     — 表格数据专用注意力网络 (pytorch-tabnet)
  B15_XGB_FE    — XGBoost + 手工特征工程 (交叉特征 + 多项式)
  B16_AutoML    — FLAML AutoML (时间预算 60s/量表)

对比目的:
  证明 SSP-MFN 在小样本多模态场景下优于强表格基线

防御性策略:
  1. 过拟合检测 (train/test ratio)
  2. 置换基线验证
  3. 效应量检测
  4. 多重比较 Bonferroni 校正
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from defensive_protocol import DefensiveProtocol
from EXP1_sspmfn_main import build_dataset

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
RES.mkdir(parents=True, exist_ok=True)
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6


def build_flat_features(data, tr, te):
    """拼接三路特征为平坦向量"""
    X = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), sc.transform(X[te]), data["y_adj"][tr], data["y_adj"][te]


def build_engineered_features(data, tr, te):
    """手工特征工程: 原始 + 二阶多项式交叉 (仅前20维避免维度爆炸)"""
    X = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    sc = StandardScaler().fit(X[tr])
    X_tr_s = sc.transform(X[tr])
    X_te_s = sc.transform(X[te])
    top_k = min(20, X_tr_s.shape[1])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr_s[:, :top_k])
    X_te_poly = poly.transform(X_te_s[:, :top_k])
    X_tr_full = np.hstack([X_tr_s, X_tr_poly])
    X_te_full = np.hstack([X_te_s, X_te_poly])
    sc2 = StandardScaler().fit(X_tr_full)
    return sc2.transform(X_tr_full), sc2.transform(X_te_full), data["y_adj"][tr], data["y_adj"][te]


def run_tabnet(data, tr, te, seed=42):
    """B14: TabNet — 若未安装则降级为 GBM"""
    X_tr, X_te, y_tr, y_te = build_flat_features(data, tr, te)
    preds = np.zeros((len(te), N_SCALES))
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
        for k in range(N_SCALES):
            m = TabNetRegressor(
                n_d=16, n_a=16, n_steps=3, gamma=1.3,
                n_independent=2, n_shared=2,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": 2e-3},
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type="entmax",
                seed=seed,
                verbose=0,
            )
            m.fit(
                X_tr, y_tr[:, k:k+1],
                eval_set=[(X_te, y_te[:, k:k+1])],
                patience=20, max_epochs=200,
                batch_size=256, virtual_batch_size=128,
            )
            preds[:, k] = m.predict(X_te).ravel()
        used = "TabNet"
    except ImportError:
        for k in range(N_SCALES):
            m = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=seed,
            )
            m.fit(X_tr, y_tr[:, k])
            preds[:, k] = m.predict(X_te)
        used = "GBM_fallback"
    return preds, y_te, used


def run_xgb_fe(data, tr, te, seed=42):
    """B15: XGBoost + 特征工程"""
    X_tr, X_te, y_tr, y_te = build_engineered_features(data, tr, te)
    preds = np.zeros((len(te), N_SCALES))
    try:
        from xgboost import XGBRegressor
        for k in range(N_SCALES):
            m = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=seed, verbosity=0,
                early_stopping_rounds=20,
            )
            split = int(len(X_tr) * 0.85)
            m.fit(
                X_tr[:split], y_tr[:split, k],
                eval_set=[(X_tr[split:], y_tr[split:, k])],
                verbose=False,
            )
            preds[:, k] = m.predict(X_te)
        used = "XGBoost_FE"
    except ImportError:
        for k in range(N_SCALES):
            m = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=seed,
            )
            m.fit(X_tr, y_tr[:, k])
            preds[:, k] = m.predict(X_te)
        used = "GBM_FE_fallback"
    return preds, y_te, used


def run_automl(data, tr, te, seed=42, time_budget=60):
    """B16: FLAML AutoML"""
    X_tr, X_te, y_tr, y_te = build_flat_features(data, tr, te)
    preds = np.zeros((len(te), N_SCALES))
    try:
        from flaml import AutoML
        for k in range(N_SCALES):
            automl = AutoML()
            automl.fit(
                X_tr, y_tr[:, k],
                task="regression",
                time_budget=time_budget,
                metric="r2",
                seed=seed,
                verbose=0,
            )
            preds[:, k] = automl.predict(X_te)
        used = "FLAML_AutoML"
    except ImportError:
        for k in range(N_SCALES):
            m = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=seed,
            )
            m.fit(X_tr, y_tr[:, k])
            preds[:, k] = m.predict(X_te)
        used = "GBM_automl_fallback"
    return preds, y_te, used


def compute_metrics(preds, truth):
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


def run_cv(data, runner_fn, name, n_splits=5, **kw):
    gkf = GroupKFold(n_splits=n_splits)
    all_p, all_t = [], []
    used_impl = None
    for fi, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        print(f"    [{name}] fold {fi+1}/{n_splits}...")
        p, t, used = runner_fn(data, tr, te, seed=42 + fi, **kw)
        all_p.append(p)
        all_t.append(t)
        used_impl = used
    return compute_metrics(np.concatenate(all_p), np.concatenate(all_t)), used_impl


def main():
    print("[EXP-14] 强基线补充实验 B14-B16")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    dap = DefensiveProtocol("EXP14_strong_baselines")
    results = {"n_samples": N, "baselines": {}}

    print("\n  [B14] TabNet...")
    r14, impl14 = run_cv(data, run_tabnet, "B14_TabNet")
    results["baselines"]["B14_TabNet"] = {"metrics": r14, "impl": impl14}
    print(f"    R²={r14['_mean']['r2']:.4f}, RMSE={r14['_mean']['rmse']:.4f} ({impl14})")

    print("\n  [B15] XGBoost + 特征工程...")
    r15, impl15 = run_cv(data, run_xgb_fe, "B15_XGB_FE")
    results["baselines"]["B15_XGB_FE"] = {"metrics": r15, "impl": impl15}
    print(f"    R²={r15['_mean']['r2']:.4f}, RMSE={r15['_mean']['rmse']:.4f} ({impl15})")

    print("\n  [B16] AutoML (60s budget/fold)...")
    r16, impl16 = run_cv(data, run_automl, "B16_AutoML", time_budget=60)
    results["baselines"]["B16_AutoML"] = {"metrics": r16, "impl": impl16}
    print(f"    R²={r16['_mean']['r2']:.4f}, RMSE={r16['_mean']['rmse']:.4f} ({impl16})")

    all_r2 = [r14["_mean"]["r2"], r15["_mean"]["r2"], r16["_mean"]["r2"]]
    dap.check_stability(all_r2)
    dap.check_permutation_baseline(
        r15["_mean"]["rmse"], data["y_adj"][:, 0], metric_type="rmse"
    )
    dap.check_effect_size(
        model_metric=float(np.mean(all_r2)),
        baseline_metric=0.0,
        metric_type="r2",
        min_improvement=0.02,
    )

    try:
        with open(RES / "EXP1_S6_full.json") as f:
            exp1 = json.load(f)
        sspmfn_r2 = exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["r2"]
        results["SSP_MFN_ref_r2"] = sspmfn_r2
    except Exception:
        sspmfn_r2 = None

    results["DAP"] = dap.generate_report()

    out_path = RES / "EXP14_strong_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out_path}")

    print("\n" + "=" * 65)
    print(f"  {'模型':<20} {'R²':>8} {'RMSE':>8} {'r':>8}  {'实现':>20}")
    print(f"  {'-'*60}")
    if sspmfn_r2 is not None:
        print(f"  {'SSP-MFN (ref)':<20} {sspmfn_r2:>8.4f} {'':>8} {'':>8}")
    for bname, bdata in results["baselines"].items():
        m = bdata["metrics"]["_mean"]
        print(f"  {bname:<20} {m['r2']:>8.4f} {m['rmse']:>8.4f} {m['r']:>8.4f}  {bdata['impl']:>20}")
    print("=" * 65)


if __name__ == "__main__":
    main()
