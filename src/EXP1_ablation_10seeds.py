"""EXP-1b · SSP-MFN 消融 × 10 种子（方案A）

目的：修复 Table 2 vs Table 5 AdaIN 方向反转问题。
原因：原脚本消融变体仅用单种子，Full 模型用 best-of-3，协议不一致。
本脚本对 4 个消融变体全部用相同的 10 种子，报告 mean±SD。

输出：实验/results/EXP1_ablation_10seeds.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from EXP1_sspmfn_main import build_dataset, train_sspmfn, run_cv, DEVICE

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)

SEEDS = [17, 42, 2024, 0, 7, 123, 456, 789, 1000, 2025]

VARIANTS = {
    "Full":     dict(use_gate=True,  use_adain=True),
    "No-Gate":  dict(use_gate=False, use_adain=True),
    "No-AdaIN": dict(use_gate=True,  use_adain=False),
    "Plain":    dict(use_gate=False, use_adain=False),
}


def run_variant_multiseed(data, variant_kwargs: dict, seeds: list[int]) -> dict:
    r2_list, rmse_list = [], []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        cv_result = run_cv(data, train_sspmfn, **variant_kwargs)
        r2_list.append(cv_result["_mean"]["r2"])
        rmse_list.append(cv_result["_mean"]["rmse"])
    return {
        "r2_mean":   round(float(np.mean(r2_list)),  4),
        "r2_std":    round(float(np.std(r2_list)),   4),
        "rmse_mean": round(float(np.mean(rmse_list)), 4),
        "rmse_std":  round(float(np.std(rmse_list)),  4),
        "r2_per_seed":   [round(x, 4) for x in r2_list],
        "rmse_per_seed": [round(x, 4) for x in rmse_list],
        "seeds": seeds,
    }


def main():
    print(f"[EXP-1b] device={DEVICE}, 10 seeds × 4 variants")
    data = build_dataset(frac=1.0, seed=42)
    print(f"  N={len(data['y_adj'])}")

    results = {}
    for name, kwargs in VARIANTS.items():
        print(f"\n  [{name}] running {len(SEEDS)} seeds...")
        results[name] = run_variant_multiseed(data, kwargs, SEEDS)
        r = results[name]
        print(f"    R² = {r['r2_mean']:.4f} ± {r['r2_std']:.4f}  "
              f"RMSE = {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")

    # 计算 ΔR² (相对 Full)
    full_r2 = results["Full"]["r2_mean"]
    print("\n  ΔR² relative to Full (mean):")
    for name, r in results.items():
        delta = r["r2_mean"] - full_r2
        print(f"    {name:<12}: ΔR² = {delta:+.4f}  "
              f"(Full={full_r2:.4f}, {name}={r['r2_mean']:.4f})")
        results[name]["delta_r2_vs_full"] = round(delta, 4)

    out_path = RES / "EXP1_ablation_10seeds.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved → {out_path}")

    print("\n" + "="*60)
    print(f"  {'Variant':<12} {'R² mean':<10} {'R² std':<10} {'ΔR²':<10}")
    print(f"  {'-'*50}")
    for name, r in results.items():
        print(f"  {name:<12} {r['r2_mean']:<10.4f} {r['r2_std']:<10.4f} "
              f"{r['delta_r2_vs_full']:+.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
