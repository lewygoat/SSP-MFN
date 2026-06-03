"""EXP-19 · IRI 与 SCI-2 量表的项目-总分相关 (item-total) 分析

对应论文 K8 返修意见: 已知 IRI α=0.169, SCI-2 α=0.174,
逐项剔除查看 α 是否回升, 输出补充表 S4 用于支撑量表局限性讨论.

计算口径:
  - 整体 α (Cronbach): 全部 6 项
  - 校正后项总相关 (corrected item-total r): 每项与其他5项之和的 Pearson r
  - α-if-item-deleted: 删除某项后剩余 5 项的 α
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"


def cronbach_alpha(X: np.ndarray) -> float:
    n_items = X.shape[1]
    if n_items < 2:
        return float("nan")
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return (n_items / (n_items - 1.0)) * (1.0 - item_var.sum() / total_var)


def corrected_item_total(X: np.ndarray, idx: int) -> float:
    item = X[:, idx]
    rest = np.delete(X, idx, axis=1).sum(axis=1)
    if item.std() == 0 or rest.std() == 0:
        return float("nan")
    return float(np.corrcoef(item, rest)[0, 1])


def analyse(name: str, items: list, df: pd.DataFrame):
    X = df[items].dropna().to_numpy(dtype=float)
    n_resp = X.shape[0]
    full_alpha = cronbach_alpha(X)
    print(f"\n[{name}] n={n_resp}, α(6 items)={full_alpha:.4f}")

    rows = []
    for i, it in enumerate(items):
        rest_X = np.delete(X, i, axis=1)
        a_del = cronbach_alpha(rest_X)
        r_total = corrected_item_total(X, i)
        rows.append({
            "item": it,
            "mean": float(X[:, i].mean()),
            "std": float(X[:, i].std(ddof=1)),
            "corrected_item_total_r": round(r_total, 4),
            "alpha_if_deleted": round(a_del, 4),
        })
        print(f"  {it}: mean={X[:, i].mean():.3f}, r_it={r_total:+.4f}, α_del={a_del:.4f}")

    sorted_rows = sorted(rows, key=lambda r: r["alpha_if_deleted"], reverse=True)
    best = sorted_rows[0]
    print(f"  → best deletion: drop {best['item']} → α={best['alpha_if_deleted']:.4f} (Δ={best['alpha_if_deleted']-full_alpha:+.4f})")

    rest_items = [it for it in items if it != best["item"]]
    X5 = df[rest_items].dropna().to_numpy(dtype=float)
    a5 = cronbach_alpha(X5)
    print(f"  recompute 5-item α (after dropping {best['item']}): {a5:.4f}")

    return {
        "n": n_resp,
        "alpha_6": round(full_alpha, 4),
        "per_item": rows,
        "best_deletion": {
            "item": best["item"],
            "new_alpha": round(best["alpha_if_deleted"], 4),
            "delta_alpha": round(best["alpha_if_deleted"] - full_alpha, 4),
        },
        "five_item_alpha_after_drop": round(a5, 4),
    }


def main():
    df_all = pd.read_csv(ROOT / "数据/数据v2/scale_table_v2.csv")
    print(f"loaded n={len(df_all)} observations from scale_table_v2")
    print("timepoints:", df_all["timepoint"].value_counts().to_dict())

    iri_items = [f"iri_{i}" for i in range(1, 7)]
    sci2_items = [f"sci2_{i}" for i in range(1, 7)]

    out = {"by_timepoint": {}}
    for tp in ["pre", "post", "delayed", "all"]:
        df = df_all if tp == "all" else df_all[df_all["timepoint"] == tp]
        print(f"\n========== timepoint = {tp} (n={len(df)}) ==========")
        out["by_timepoint"][tp] = {
            "IRI": analyse(f"IRI ({tp})", iri_items, df),
            "SCI2": analyse(f"SCI-2 ({tp})", sci2_items, df),
        }

    out_path = RES / "EXP19_item_total_iri_sci2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
