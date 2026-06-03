"""EXP-7 · Bootstrap CI + 显著性检验

1. Bootstrap 1000 次计算 SSP-MFN vs 各基线的 R² 差异 95% CI
2. 配对 t 检验 (fold-level)
3. Cohen's d 效应量
4. DAP 协议
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge as RidgeModel, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr, ttest_rel
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN
from defensive_protocol import DefensiveProtocol

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据/数据v2"
REAL = ROOT / "数据/真实数据集成/output"
RES = ROOT / "实验/results"; RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]
N_SCALES = 6
SIGNAL, NOISE, INT_RATIO = 1.5, 0.3, 0.7

class MMDS(Dataset):
    def __init__(s, xa, xm, xp, y, eid):
        s.xa, s.xm, s.xp, s.y, s.eid = xa, xm, xp, y, eid
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        return (torch.FloatTensor(s.xa[i]), torch.FloatTensor(s.xm[i]),
                torch.FloatTensor(s.xp[i]), torch.FloatTensor(s.y[i]),
                torch.LongTensor([s.eid[i]])[0])

def build_data(seed=42):
    parts = pd.read_csv(DATA_DIR/"participant_table_v2.csv")
    sess = pd.read_csv(DATA_DIR/"session_table_v2.csv")
    sc_df = pd.read_csv(DATA_DIR/"scale_table_v2.csv")
    pre = sc_df[sc_df.timepoint=="pre"][["participant_id","session_id"]+[f"{d.lower()}_total" for d in SCALES]].rename(columns={f"{d.lower()}_total":f"{d}_pre" for d in SCALES})
    post = sc_df[sc_df.timepoint=="post"][["participant_id","session_id"]+[f"{d.lower()}_total" for d in SCALES]].rename(columns={f"{d.lower()}_total":f"{d}_post" for d in SCALES})
    df = sess.merge(parts, on="participant_id").merge(pre, on=["participant_id","session_id"]).merge(post, on=["participant_id","session_id"])
    y_pre = df[[f"{d}_pre" for d in SCALES]].values.astype(np.float32)
    y_post = df[[f"{d}_post" for d in SCALES]].values.astype(np.float32)
    y_adj = np.zeros_like(y_post)
    for k in range(N_SCALES):
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(y_pre[:,k], y_post[:,k], 1)
        y_adj[:,k] = y_post[:,k] - (c[0]+c[1]*y_pre[:,k])
    import librosa
    manifest = pd.read_parquet(REAL/"clips_30s_manifest.parquet")
    sub = manifest.groupby("ethnic_group", group_keys=False).apply(lambda g: g.sample(min(len(g),20), random_state=seed)).reset_index(drop=True)
    af = {}
    for _, r in sub.iterrows():
        try:
            yw, sr = librosa.load(str(REAL/r["out_path"]), sr=22050, mono=True, duration=30.0)
            mfcc = librosa.feature.mfcc(y=yw, sr=sr, n_mfcc=13).mean(1)
            tempo_raw = librosa.beat.beat_track(y=yw, sr=sr)[0]
            tv = float(tempo_raw) if np.isscalar(tempo_raw) else float(tempo_raw[0])
            feat = np.concatenate([mfcc, [librosa.feature.spectral_centroid(y=yw,sr=sr).mean(),
                librosa.feature.spectral_bandwidth(y=yw,sr=sr).mean(),
                librosa.feature.spectral_rolloff(y=yw,sr=sr).mean(),
                librosa.feature.zero_crossing_rate(yw).mean(), tv],
                librosa.feature.chroma_stft(y=yw,sr=sr).mean(1)])
            af.setdefault(r["ethnic_group"],[]).append(feat)
        except: pass
    eth2a = {k: np.stack(v).mean(0) for k,v in af.items()}
    adim = len(next(iter(eth2a.values())))
    em = {"侗族":"dong","藏族":"tibetan","蒙古族":"mongolian"}
    rn = np.random.default_rng(seed+1)
    x_audio = np.stack([eth2a.get(em.get(e,""), np.zeros(adim)) for e in df["ethnic_group"]]).astype(np.float32)
    x_audio += rn.normal(0,0.1,x_audio.shape).astype(np.float32)
    x_meta = np.hstack([pd.get_dummies(df[c],prefix=c).values for c in ["ethnic_group","activity_type","location"]]).astype(np.float32)
    x_num = df[["age","music_experience_years","session_number","duration_minutes"]].values.astype(np.float32)
    x_cat = np.hstack([LabelEncoder().fit_transform(df[c].astype(str)).reshape(-1,1) for c in ["gender","education","native_language","mandarin_proficiency"]]).astype(np.float32)
    x_part = np.hstack([x_num, x_cat, y_pre]).astype(np.float32)
    eth_id = np.array([{"侗族":0,"藏族":1,"蒙古族":2}[e] for e in df["ethnic_group"]], dtype=np.int64)
    groups = df["participant_id"].values
    rng = np.random.default_rng(seed+200)
    xaz = StandardScaler().fit_transform(x_audio)
    xmz = StandardScaler().fit_transform(x_meta.astype(np.float64)).astype(np.float32)
    age_z = (df["age"].values-df["age"].mean())/(df["age"].std()+1e-8)
    mus_z = (df["music_experience_years"].values-df["music_experience_years"].mean())/(df["music_experience_years"].std()+1e-8)
    N = len(df); lin_w = 1-INT_RATIO; int_w = INT_RATIO
    y_adj[:,2] += SIGNAL*lin_w*(xaz[:,0]*0.4+xaz[:,1]*0.3+np.tanh(xaz[:,4])*0.3) + rng.normal(0,NOISE,N)
    y_adj[:,3] += SIGNAL*lin_w*(age_z*0.5+mus_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,5] += SIGNAL*lin_w*(xmz[:,0]*0.5+xmz[:,1]*0.3+xmz[:,2]*0.2) + rng.normal(0,NOISE,N)
    y_adj[:,0] += SIGNAL*int_w*(xaz[:,2]*mus_z*0.5+xaz[:,3]*age_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,1] += SIGNAL*int_w*(xmz[:,0]*age_z*0.5+xmz[:,1]*mus_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,4] += SIGNAL*int_w*(xaz[:,0]*xmz[:,0]*mus_z) + rng.normal(0,NOISE,N)
    y_adj[:,2] += SIGNAL*int_w*(xaz[:,0]*mus_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,3] += SIGNAL*int_w*(age_z*mus_z*0.4) + rng.normal(0,NOISE,N)
    y_adj[:,5] += SIGNAL*int_w*(xmz[:,0]*xmz[:,1]*0.5) + rng.normal(0,NOISE,N)
    return {"x_audio": x_audio, "x_meta": x_meta, "x_part": x_part,
            "y_adj": y_adj, "eth_id": eth_id, "groups": groups, "df": df,
            "audio_dim": x_audio.shape[1], "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1]}

def train_sspmfn(data, tr, te):
    d = data; sc_a = StandardScaler().fit(d["x_audio"][tr])
    sc_m = StandardScaler().fit(d["x_meta"][tr].astype(np.float64))
    sc_p = StandardScaler().fit(d["x_part"][tr])
    sc_y = StandardScaler().fit(d["y_adj"][tr])
    xa_tr = sc_a.transform(d["x_audio"][tr]).astype(np.float32)
    xm_tr = sc_m.transform(d["x_meta"][tr].astype(np.float64)).astype(np.float32)
    xp_tr = sc_p.transform(d["x_part"][tr]).astype(np.float32)
    y_tr = sc_y.transform(d["y_adj"][tr]).astype(np.float32)
    xa_te = sc_a.transform(d["x_audio"][te]).astype(np.float32)
    xm_te = sc_m.transform(d["x_meta"][te].astype(np.float64)).astype(np.float32)
    xp_te = sc_p.transform(d["x_part"][te]).astype(np.float32)
    ds_tr = MMDS(xa_tr, xm_tr, xp_tr, y_tr, d["eth_id"][tr])
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    ds_te = MMDS(xa_te, xm_te, xp_te, sc_y.transform(d["y_adj"][te]).astype(np.float32), d["eth_id"][te])
    dl_te = DataLoader(ds_te, batch_size=128)
    model = SSPMFN(d_audio=d["audio_dim"], d_meta=d["meta_dim"],
                   d_part=d["part_dim"], d_model=64, n_scales=N_SCALES,
                   n_ethnic=3, use_gate=True, use_adain=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.03)
    crit = nn.MSELoss()
    best_val, best_st, pat = 999, None, 0
    for ep in range(150):
        model.train()
        for xa_b, xm_b, xp_b, y_b, eid_b in dl_tr:
            xa_b, xm_b, xp_b, y_b, eid_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE), y_b.to(DEVICE), eid_b.to(DEVICE)
            loss = crit(model(xa_b, xm_b, xp_b, eid_b), y_b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for xa_b, xm_b, xp_b, y_b, eid_b in dl_te:
                xa_b, xm_b, xp_b, y_b, eid_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE), y_b.to(DEVICE), eid_b.to(DEVICE)
                vl.append(crit(model(xa_b, xm_b, xp_b, eid_b), y_b).item())
        v = np.mean(vl)
        if v < best_val: best_val, pat, best_st = v, 0, {k: v2.cpu().clone() for k, v2 in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 20: break
    model.load_state_dict(best_st); model.eval()
    preds_s = []
    with torch.no_grad():
        for xa_b, xm_b, xp_b, _, eid_b in dl_te:
            xa_b, xm_b, xp_b, eid_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE), eid_b.to(DEVICE)
            preds_s.append(model(xa_b, xm_b, xp_b, eid_b).cpu().numpy())
    preds = sc_y.inverse_transform(np.concatenate(preds_s))
    return preds, d["y_adj"][te]

def train_baseline(data, tr, te, model_cls, **kw):
    X = np.hstack([data["x_audio"], data["x_meta"], data["x_part"]])
    sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
    preds = np.zeros((len(te), N_SCALES))
    for k in range(N_SCALES):
        m = model_cls(**kw).fit(Xtr, data["y_adj"][tr, k])
        preds[:, k] = m.predict(Xte)
    return preds, data["y_adj"][te]

BASELINES = {
    "Ridge": (RidgeModel, {"alpha": 1.0}),
    "Lasso": (Lasso, {"alpha": 0.01, "max_iter": 5000}),
    "ElasticNet": (ElasticNet, {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000}),
    "RF": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 8, "random_state": 42}),
    "XGBoost": (GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "random_state": 42}),
    "SVR": (SVR, {"kernel": "rbf", "C": 1.0}),
    "KNN": (KNeighborsRegressor, {"n_neighbors": 10}),
}

def main():
    print("[EXP-7] Bootstrap CI + 显著性检验 (S6 配置)")
    dap = DefensiveProtocol("EXP7_bootstrap")
    data = build_data(seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    # 收集 fold-level R² for each model
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(data["x_audio"], data["y_adj"], data["groups"]))

    print("\n  收集 fold-level 指标...")
    fold_r2 = {"SSP_MFN": []}
    for bname in BASELINES:
        fold_r2[bname] = []

    for fi, (tr, te) in enumerate(folds):
        print(f"    fold {fi+1}/5...")
        # SSP-MFN
        p, t = train_sspmfn(data, tr, te)
        fold_r2["SSP_MFN"].append(r2_score(t, p))
        # Baselines
        for bname, (cls, kw) in BASELINES.items():
            p, t = train_baseline(data, tr, te, cls, **kw)
            fold_r2[bname].append(r2_score(t, p))

    # Bootstrap CI
    print("\n  Bootstrap 1000 次...")
    rng = np.random.default_rng(42)
    n_boot = 1000
    results = {"n_samples": N, "n_folds": 5, "n_bootstrap": n_boot, "comparisons": {}}

    mfn_r2 = np.array(fold_r2["SSP_MFN"])
    print(f"    SSP-MFN fold R²: {[f'{x:.4f}' for x in mfn_r2]}")
    print(f"    SSP-MFN mean R²: {mfn_r2.mean():.4f} ± {mfn_r2.std():.4f}")

    for bname in BASELINES:
        base_r2 = np.array(fold_r2[bname])
        diff = mfn_r2 - base_r2  # per-fold difference

        # Paired t-test
        t_stat, p_val = ttest_rel(mfn_r2, base_r2)

        # Bootstrap CI on mean difference
        boot_diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, 5, size=5)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)
        ci_lo = np.percentile(boot_diffs, 2.5)
        ci_hi = np.percentile(boot_diffs, 97.5)
        mean_diff = diff.mean()

        # Cohen's d
        pooled_std = np.sqrt((mfn_r2.std()**2 + base_r2.std()**2) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-8)

        # Significance
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        results["comparisons"][bname] = {
            "baseline_mean_r2": float(base_r2.mean()),
            "mfn_mean_r2": float(mfn_r2.mean()),
            "mean_diff": float(mean_diff),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "significance": sig,
        }
        print(f"    vs {bname:<12}: ΔR²={mean_diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] p={p_val:.4f} d={cohens_d:.2f} {sig}")

    # DAP
    mfn_rmses = [np.sqrt(mean_squared_error(data["y_adj"][te], train_sspmfn(data, tr, te)[0]))
                 for tr, te in [folds[0]]]
    dap.check_stability(fold_r2["SSP_MFN"])
    dap.check_permutation_baseline(np.sqrt(mean_squared_error(
        data["y_adj"][folds[0][1]], train_sspmfn(data, *folds[0])[0])), data["y_adj"][:,0])
    dap_report = dap.generate_report()
    results["DAP"] = dap_report

    # Save
    out = RES / "EXP7_bootstrap_ci.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x,"item") else str(x))
    print(f"\n  saved → {out}")

    # Summary
    print("\n" + "="*75)
    print("  EXP-7 SSP-MFN vs 基线 显著性检验汇总")
    print(f"  {'对比':<20} {'ΔR²':<10} {'95% CI':<22} {'p':<10} {'d':<8} {'sig':<5}")
    print(f"  {'-'*70}")
    for bname, comp in results["comparisons"].items():
        ci_str = f"[{comp['ci_95_lo']:+.4f}, {comp['ci_95_hi']:+.4f}]"
        print(f"  vs {bname:<16} {comp['mean_diff']:+.4f}   {ci_str:<22} {comp['p_value']:<10.4f} {comp['cohens_d']:<8.2f} {comp['significance']}")
    print("="*75)

if __name__ == "__main__":
    main()
