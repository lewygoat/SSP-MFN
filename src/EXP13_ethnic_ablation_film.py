"""EXP-13 · 去 ethnic 消融 + FiLM 替代 AdaIN (R1-5, R2-2)

实验设计:
  A) 消融 ethnic 条件化:
     - SSP-MFN (full):    use_adain=True  (基准)
     - No-AdaIN:          use_adain=False (去掉 Cultural AdaIN)
     - No-Gate:           use_gate=False  (去掉门控，均值融合)
     - No-AdaIN-No-Gate:  两者都去掉

  B) FiLM 替代 AdaIN:
     - FiLM 条件化: γ/β 由 ethnic embedding 生成，但作用于 feature-wise 仿射
       (与 AdaIN 区别: FiLM 不做实例归一化，直接仿射变换)

报告:
  - 5折 CV R²/RMSE/r (逐量表 + 均值)
  - AdaIN vs FiLM vs No-ethnic 三路对比
  - 防御性: 过拟合检测 + 置换基线

防御性策略:
  1. 过拟合检测 (train/test loss ratio)
  2. 置换基线验证
  3. 多重比较校正
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN, ModalProjector, GatedFusionLayer
from defensive_protocol import DefensiveProtocol
from EXP1_sspmfn_main import build_dataset, MultiModalDS

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
N_SCALES = 6


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: 直接仿射，不做实例归一化"""
    def __init__(self, feat_dim: int, n_ethnic: int = 3, emb_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_ethnic, emb_dim)
        self.to_affine = nn.Linear(emb_dim, feat_dim * 2)

    def forward(self, h, ethnic_id):
        e = self.embed(ethnic_id)
        gamma_beta = self.to_affine(e)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class SSPMFNWithFiLM(nn.Module):
    """SSP-MFN 变体: 用 FiLM 替换 AdaIN"""
    def __init__(self, d_audio, d_meta, d_part, d_model=64,
                 n_ethnic=3, n_scales=6, p_drop=0.2):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.proj_audio = ModalProjector(d_audio, d_model, p_drop)
        self.proj_meta = ModalProjector(d_meta, d_model, p_drop)
        self.proj_part = ModalProjector(d_part, d_model, p_drop)
        self.fusion = GatedFusionLayer(d_model, n_modalities=3, n_scales=n_scales)
        self.film = FiLMLayer(d_model, n_ethnic=n_ethnic, emb_dim=8)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_scales)
        ])

    def forward(self, x_audio, x_meta, x_part, ethnic_id, **kwargs):
        z_list = [self.proj_audio(x_audio), self.proj_meta(x_meta), self.proj_part(x_part)]
        f = self.fusion(z_list)
        preds = []
        for k in range(self.n_scales):
            fk = self.film(f[:, k, :], ethnic_id)
            preds.append(self.heads[k](fk))
        return torch.cat(preds, dim=-1)


def train_fold(data, tr, te, model_factory, seed=42, epochs=200, patience=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    sc_a = StandardScaler().fit(data["x_audio"][tr])
    sc_m = StandardScaler().fit(data["x_meta"][tr])
    sc_p = StandardScaler().fit(data["x_part"][tr])
    sc_y = StandardScaler().fit(data["y_adj"][tr])

    def prep(idx):
        return (
            sc_a.transform(data["x_audio"][idx]).astype(np.float32),
            sc_m.transform(data["x_meta"][idx]).astype(np.float32),
            sc_p.transform(data["x_part"][idx]).astype(np.float32),
        )

    xa_tr, xm_tr, xp_tr = prep(tr)
    xa_te, xm_te, xp_te = prep(te)
    y_tr_s = sc_y.transform(data["y_adj"][tr]).astype(np.float32)

    tr_ds = MultiModalDS(xa_tr, xm_tr, xp_tr, y_tr_s, data["eth_id"][tr])
    te_ds = MultiModalDS(xa_te, xm_te, xp_te,
                         sc_y.transform(data["y_adj"][te]).astype(np.float32),
                         data["eth_id"][te])
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = model_factory().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    crit = nn.HuberLoss(delta=1.0)

    best_val, best_st, pat = float("inf"), None, 0
    train_losses = []
    for _ in range(epochs):
        model.train()
        ep_loss = []
        for xa_b, xm_b, xp_b, y_b, eid_b in tr_dl:
            xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
            y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xa_b, xm_b, xp_b, eid_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(loss.item())
        train_losses.append(np.mean(ep_loss))
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
    final_train_loss = float(np.mean(train_losses[-5:]))
    return preds, data["y_adj"][te], final_train_loss, float(best_val)


def run_cv(data, model_factory, name: str, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    all_p, all_t = [], []
    train_losses, test_losses = [], []
    for fi, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        print(f"    [{name}] fold {fi+1}/{n_splits}...")
        p, t, trl, tel = train_fold(data, tr, te, model_factory, seed=42 + fi)
        all_p.append(p)
        all_t.append(t)
        train_losses.append(trl)
        test_losses.append(tel)
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)
    metrics = {}
    for k, s in enumerate(SCALES):
        rmse = float(np.sqrt(mean_squared_error(all_t[:, k], all_p[:, k])))
        r2 = float(r2_score(all_t[:, k], all_p[:, k]))
        r, _ = pearsonr(all_t[:, k], all_p[:, k])
        metrics[s] = {"rmse": round(rmse, 4), "r2": round(r2, 4), "r": round(float(r), 4)}
    metrics["_mean"] = {
        "rmse": round(float(np.mean([metrics[s]["rmse"] for s in SCALES])), 4),
        "r2": round(float(np.mean([metrics[s]["r2"] for s in SCALES])), 4),
        "r": round(float(np.mean([metrics[s]["r"] for s in SCALES])), 4),
    }
    return metrics, float(np.mean(train_losses)), float(np.mean(test_losses))


def main():
    print("[EXP-13] 去 ethnic 消融 + FiLM 替代 AdaIN")
    data = build_dataset(frac=1.0, seed=42)
    N = len(data["y_adj"])
    d_a, d_m, d_p = data["audio_dim"], data["meta_dim"], data["part_dim"]
    print(f"  N={N}, audio={d_a}, meta={d_m}, part={d_p}")

    configs = {
        "SSP_MFN_full": lambda: SSPMFN(d_a, d_m, d_p, 64, 3, N_SCALES, use_gate=True, use_adain=True),
        "No_AdaIN": lambda: SSPMFN(d_a, d_m, d_p, 64, 3, N_SCALES, use_gate=True, use_adain=False),
        "No_Gate": lambda: SSPMFN(d_a, d_m, d_p, 64, 3, N_SCALES, use_gate=False, use_adain=True),
        "No_AdaIN_No_Gate": lambda: SSPMFN(d_a, d_m, d_p, 64, 3, N_SCALES, use_gate=False, use_adain=False),
        "FiLM_replace": lambda: SSPMFNWithFiLM(d_a, d_m, d_p, 64, 3, N_SCALES),
    }

    dap = DefensiveProtocol("EXP13_ethnic_ablation")
    results = {"n_samples": N, "variants": {}}
    all_r2 = []

    for name, factory in configs.items():
        print(f"\n  运行 {name}...")
        metrics, trl, tel = run_cv(data, factory, name)
        results["variants"][name] = {"metrics": metrics, "train_loss": trl, "test_loss": tel}
        all_r2.append(metrics["_mean"]["r2"])
        print(f"    R²={metrics['_mean']['r2']:.4f}, RMSE={metrics['_mean']['rmse']:.4f}")
        dap.check_overfit(trl, tel)

    dap.check_stability(all_r2)
    dap.check_permutation_baseline(
        results["variants"]["SSP_MFN_full"]["metrics"]["_mean"]["rmse"],
        data["y_adj"][:, 0],
        metric_type="rmse",
    )
    p_vals = []
    from scipy.stats import ttest_1samp
    for name in configs:
        r2_val = results["variants"][name]["metrics"]["_mean"]["r2"]
        _, p = ttest_1samp([r2_val], 0.0)
        p_vals.append(float(p))
    dap.check_multiple_comparisons(p_vals, method="bonferroni")

    results["DAP"] = dap.generate_report()

    out_path = RES / "EXP13_ethnic_ablation_film.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out_path}")

    print("\n" + "=" * 70)
    print(f"  {'变体':<22} {'R²':>8} {'RMSE':>8} {'r':>8}  {'ΔR² vs full':>12}")
    print(f"  {'-'*60}")
    full_r2 = results["variants"]["SSP_MFN_full"]["metrics"]["_mean"]["r2"]
    for name in configs:
        m = results["variants"][name]["metrics"]["_mean"]
        delta = m["r2"] - full_r2
        marker = "" if name == "SSP_MFN_full" else f"{delta:+.4f}"
        print(f"  {name:<22} {m['r2']:>8.4f} {m['rmse']:>8.4f} {m['r']:>8.4f}  {marker:>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
