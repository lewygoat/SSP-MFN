"""EXP-2 · 2³=8 模态消融矩阵 (S6 配置)

对 audio/meta/part 三路模态的所有组合进行消融:
000(无模态) 001 010 011 100 101 110 111(全模态)
逐维度报告 6 个量表的 R²/RMSE/r
集成 DAP 防御性分析协议
"""
import json, sys, warnings
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr
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
    # S6 信号
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
            "y_adj": y_adj, "eth_id": eth_id, "groups": groups,
            "audio_dim": x_audio.shape[1], "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1]}

def train_sspmfn_masked(data, tr, te, use_audio=True, use_meta=True, use_part=True):
    """训练 SSP-MFN，可选择性 mask 模态"""
    xa_tr, xm_tr, xp_tr = data["x_audio"][tr], data["x_meta"][tr], data["x_part"][tr]
    xa_te, xm_te, xp_te = data["x_audio"][te], data["x_meta"][te], data["x_part"][te]
    y_tr, y_te = data["y_adj"][tr], data["y_adj"][te]
    eid_tr, eid_te = data["eth_id"][tr], data["eth_id"][te]
    if not use_audio: xa_tr = np.zeros_like(xa_tr); xa_te = np.zeros_like(xa_te)
    if not use_meta: xm_tr = np.zeros_like(xm_tr); xm_te = np.zeros_like(xm_te)
    if not use_part: xp_tr = np.zeros_like(xp_tr); xp_te = np.zeros_like(xp_te)
    sc_a = StandardScaler().fit(xa_tr); sc_m = StandardScaler().fit(xm_tr)
    sc_p = StandardScaler().fit(xp_tr); sc_y = StandardScaler().fit(y_tr)
    xa_tr, xa_te = sc_a.transform(xa_tr), sc_a.transform(xa_te)
    xm_tr, xm_te = sc_m.transform(xm_tr), sc_m.transform(xm_te)
    xp_tr, xp_te = sc_p.transform(xp_tr), sc_p.transform(xp_te)
    y_tr_s = sc_y.transform(y_tr)
    y_te_s = sc_y.transform(y_te)
    ds_tr = MMDS(xa_tr, xm_tr, xp_tr, y_tr_s, eid_tr)
    ds_te = MMDS(xa_te, xm_te, xp_te, y_te_s, eid_te)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=128)
    model = SSPMFN(d_audio=data["audio_dim"], d_meta=data["meta_dim"],
                   d_part=data["part_dim"], d_model=64, n_scales=N_SCALES,
                   n_ethnic=3, use_gate=True, use_adain=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
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
                xa_b, xm_b, xp_b, eid_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE), eid_b.to(DEVICE)
                y_b = y_b.to(DEVICE)
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
    return preds, y_te

def run_cv(data, n_splits=5, **kwargs):
    gkf = GroupKFold(n_splits=n_splits)
    all_p, all_t = [], []
    for tr, te in gkf.split(data["x_audio"], data["y_adj"], data["groups"]):
        p, t = train_sspmfn_masked(data, tr, te, **kwargs)
        all_p.append(p); all_t.append(t)
    all_p, all_t = np.concatenate(all_p), np.concatenate(all_t)
    results = {}
    for k, name in enumerate(SCALES):
        rmse = np.sqrt(mean_squared_error(all_t[:,k], all_p[:,k]))
        r2 = r2_score(all_t[:,k], all_p[:,k])
        r, _ = pearsonr(all_t[:,k], all_p[:,k])
        results[name] = {"rmse": round(rmse,4), "r2": round(r2,4), "r": round(float(r),4)}
    results["_mean"] = {
        "rmse": round(np.mean([results[s]["rmse"] for s in SCALES]),4),
        "r2": round(np.mean([results[s]["r2"] for s in SCALES]),4),
        "r": round(np.mean([results[s]["r"] for s in SCALES]),4)}
    return results

def main():
    print("[EXP-2] 2³=8 模态消融矩阵 (S6 配置)")
    dap = DefensiveProtocol("EXP2_ablation")
    data = build_data(seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}, S6: S={SIGNAL}, σ={NOISE}, int={INT_RATIO}")

    # 8 种组合: (audio, meta, part) ∈ {0,1}³
    combos = list(product([False, True], repeat=3))
    combo_names = []
    results = {"config": "S6", "N": N}
    all_r2 = []

    print(f"\n  {'组合':<20} {'audio':>5} {'meta':>5} {'part':>5}  {'R²_mean':>8}  {'ICS':>6} {'IRI':>6} {'CSAS':>6} {'SSCS':>6} {'IOS':>6} {'SCI2':>6}")
    print(f"  {'-'*95}")

    for i, (a, m, p) in enumerate(combos):
        if not a and not m and not p:
            # 全零无意义，跳过
            continue
        name = f"{'A' if a else '_'}{'M' if m else '_'}{'P' if p else '_'}"
        combo_names.append(name)
        r = run_cv(data, use_audio=a, use_meta=m, use_part=p)
        results[name] = r
        all_r2.append(r["_mean"]["r2"])
        dims = "  ".join([f"{r[s]['r2']:>6.3f}" for s in SCALES])
        print(f"  {name:<20} {'✓' if a else '✗':>5} {'✓' if m else '✗':>5} {'✓' if p else '✗':>5}  {r['_mean']['r2']:>8.4f}  {dims}")

    # DAP 稳定性
    stab = dap.check_stability(all_r2)
    print(f"\n  [DAP] {stab.message}")

    # 计算模态边际贡献
    print("\n  === 模态边际贡献 (Marginal Contribution) ===")
    # 边际贡献 = 有该模态的组合均值 - 无该模态的组合均值
    modalities = ["audio", "meta", "part"]
    for idx, mod in enumerate(modalities):
        with_mod = [r2 for (combo, r2) in zip(combos[1:], all_r2) if combo[idx]]
        without_mod = [r2 for (combo, r2) in zip(combos[1:], all_r2) if not combo[idx]]
        margin = np.mean(with_mod) - np.mean(without_mod) if without_mod else np.mean(with_mod)
        print(f"    {mod:<10}: Δ R² = {margin:+.4f}")

    # DAP 置换基线
    perm = dap.check_permutation_baseline(
        results.get("AMP", results.get("__P", {})).get("_mean", {}).get("rmse", 1.5),
        data["y_adj"][:, 0])
    print(f"  [DAP] {perm.message}")

    report = dap.generate_report()
    results["DAP"] = report

    out = RES / "EXP2_ablation_matrix.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out}")

if __name__ == "__main__":
    main()
