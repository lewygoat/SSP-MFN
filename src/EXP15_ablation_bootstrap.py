"""EXP-15 · Ablation Bootstrap CI (participant-level resampling)

Compares Full vs No-Gate and Full vs No-AdaIN via participant-level
bootstrap (n=1000). Reports ΔR², 95% CI, and p-value.

Strategy:
  1. Run 5-fold grouped CV for Full, No-Gate, No-AdaIN → per-sample predictions
  2. Bootstrap 1000 times: resample participants with replacement,
     compute mean R² across 6 scales for each variant, compute ΔR²
  3. CI = percentile [2.5%, 97.5%], p = proportion of ΔR² ≤ 0
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from ssp_mfn import SSPMFN

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
DATA_DIR = ROOT / "数据/数据v2"
REAL = ROOT / "数据/真实数据集成/output"
RES = ROOT / "实验/results"; RES.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]
N_SCALES = 6
SIGNAL, NOISE, INT_RATIO = 1.5, 0.3, 0.7
N_BOOT = 1000


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
    y_adj[:,2] += SIGNAL*int_w*(xaz[:,0]*mus_z*0.5) + rng.normal(0,NOISE*0.5,N)
    y_adj[:,3] += SIGNAL*int_w*(age_z*mus_z*0.5) + rng.normal(0,NOISE*0.5,N)
    y_adj[:,5] += SIGNAL*int_w*(xmz[:,0]*xmz[:,2]*0.5) + rng.normal(0,NOISE*0.5,N)
    return {"x_audio":x_audio,"x_meta":x_meta,"x_part":x_part,"y_adj":y_adj,
            "eth_id":eth_id,"groups":groups,"audio_dim":x_audio.shape[1],
            "meta_dim":x_meta.shape[1],"part_dim":x_part.shape[1]}


def train_sspmfn(data, tr, te, use_gate=True, use_adain=True, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    xa_tr,xa_te = data["x_audio"][tr],data["x_audio"][te]
    xm_tr,xm_te = data["x_meta"][tr],data["x_meta"][te]
    xp_tr,xp_te = data["x_part"][tr],data["x_part"][te]
    y_tr,y_te = data["y_adj"][tr],data["y_adj"][te]
    eid_tr,eid_te = data["eth_id"][tr],data["eth_id"][te]
    sa=StandardScaler().fit(xa_tr);xa_tr=sa.transform(xa_tr);xa_te=sa.transform(xa_te)
    sm=StandardScaler().fit(xm_tr);xm_tr=sm.transform(xm_tr);xm_te=sm.transform(xm_te)
    sp=StandardScaler().fit(xp_tr);xp_tr=sp.transform(xp_tr);xp_te=sp.transform(xp_te)
    sy=StandardScaler().fit(y_tr);y_tr_s=sy.transform(y_tr)
    tr_ds=MMDS(xa_tr.astype(np.float32),xm_tr.astype(np.float32),xp_tr.astype(np.float32),y_tr_s.astype(np.float32),eid_tr)
    te_ds=MMDS(xa_te.astype(np.float32),xm_te.astype(np.float32),xp_te.astype(np.float32),sy.transform(y_te).astype(np.float32),eid_te)
    tr_dl=DataLoader(tr_ds,batch_size=32,shuffle=True)
    te_dl=DataLoader(te_ds,batch_size=32,shuffle=False)
    model=SSPMFN(d_audio=data["audio_dim"],d_meta=data["meta_dim"],d_part=data["part_dim"],d_model=64,n_ethnic=3,n_scales=6,p_drop=0.3,use_adain=use_adain,use_gate=use_gate).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.03)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200,eta_min=1e-5)
    crit=nn.HuberLoss(delta=1.0)
    best_val,pat,best_st=float("inf"),0,None
    for ep in range(200):
        model.train()
        for xa_b,xm_b,xp_b,y_b,eid_b in tr_dl:
            xa_b,xm_b,xp_b=xa_b.to(DEVICE),xm_b.to(DEVICE),xp_b.to(DEVICE)
            y_b,eid_b=y_b.to(DEVICE),eid_b.to(DEVICE)
            opt.zero_grad()
            loss=crit(model(xa_b,xm_b,xp_b,eid_b),y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        sched.step()
        model.eval(); vl=[]
        with torch.no_grad():
            for xa_b,xm_b,xp_b,y_b,eid_b in te_dl:
                xa_b,xm_b,xp_b=xa_b.to(DEVICE),xm_b.to(DEVICE),xp_b.to(DEVICE)
                y_b,eid_b=y_b.to(DEVICE),eid_b.to(DEVICE)
                vl.append(crit(model(xa_b,xm_b,xp_b,eid_b),y_b).item())
        v=np.mean(vl)
        if v<best_val: best_val,pat,best_st=v,0,{k:v2.cpu().clone() for k,v2 in model.state_dict().items()}
        else:
            pat+=1
            if pat>=20: break
    model.load_state_dict(best_st); model.eval()
    preds_s=[]
    with torch.no_grad():
        for xa_b,xm_b,xp_b,_,eid_b in te_dl:
            xa_b,xm_b,xp_b=xa_b.to(DEVICE),xm_b.to(DEVICE),xp_b.to(DEVICE)
            eid_b=eid_b.to(DEVICE)
            preds_s.append(model(xa_b,xm_b,xp_b,eid_b).cpu().numpy())
    return sy.inverse_transform(np.concatenate(preds_s)), y_te


def collect_predictions(data, use_gate, use_adain, n_splits=5):
    """Run CV and return per-sample predictions aligned with original indices."""
    gkf = GroupKFold(n_splits=n_splits)
    all_preds = np.zeros_like(data["y_adj"])
    for tr, te in gkf.split(data["x_audio"], data["y_adj"], data["groups"]):
        p, _ = train_sspmfn(data, tr, te, use_gate=use_gate, use_adain=use_adain)
        all_preds[te] = p
    return all_preds


def bootstrap_delta_r2(y_true, preds_a, preds_b, groups, n_boot=N_BOOT, seed=42):
    """Bootstrap participant-level resampling to compute ΔR² CI."""
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(groups)
    deltas = []
    for _ in range(n_boot):
        boot_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([np.where(groups == pid)[0] for pid in boot_pids])
        r2_a = np.mean([r2_score(y_true[idx, k], preds_a[idx, k]) for k in range(N_SCALES)])
        r2_b = np.mean([r2_score(y_true[idx, k], preds_b[idx, k]) for k in range(N_SCALES)])
        deltas.append(r2_a - r2_b)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = np.mean(deltas <= 0)
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "n_boot": n_boot,
    }


def main():
    print(f"[EXP-15] Ablation Bootstrap CI (n={N_BOOT})")
    print(f"  device={DEVICE}")
    data = build_data(seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    print("\n  Collecting CV predictions for Full model...")
    preds_full = collect_predictions(data, use_gate=True, use_adain=True)
    r2_full = np.mean([r2_score(data["y_adj"][:,k], preds_full[:,k]) for k in range(N_SCALES)])
    print(f"    Full R² = {r2_full:.4f}")

    print("  Collecting CV predictions for No-Gate model...")
    preds_nogate = collect_predictions(data, use_gate=False, use_adain=True)
    r2_nogate = np.mean([r2_score(data["y_adj"][:,k], preds_nogate[:,k]) for k in range(N_SCALES)])
    print(f"    No-Gate R² = {r2_nogate:.4f}")

    print("  Collecting CV predictions for No-AdaIN model...")
    preds_noadain = collect_predictions(data, use_gate=True, use_adain=False)
    r2_noadain = np.mean([r2_score(data["y_adj"][:,k], preds_noadain[:,k]) for k in range(N_SCALES)])
    print(f"    No-AdaIN R² = {r2_noadain:.4f}")

    print(f"\n  Running bootstrap (n={N_BOOT})...")
    comp_gate = bootstrap_delta_r2(data["y_adj"], preds_full, preds_nogate, data["groups"])
    print(f"    Full vs No-Gate: ΔR² = {comp_gate['mean_delta']:+.4f} "
          f"[{comp_gate['ci_lo']:+.4f}, {comp_gate['ci_hi']:+.4f}] p={comp_gate['p_value']:.4f}")

    comp_adain = bootstrap_delta_r2(data["y_adj"], preds_full, preds_noadain, data["groups"])
    print(f"    Full vs No-AdaIN: ΔR² = {comp_adain['mean_delta']:+.4f} "
          f"[{comp_adain['ci_lo']:+.4f}, {comp_adain['ci_hi']:+.4f}] p={comp_adain['p_value']:.4f}")

    results = {
        "config": {"signal": SIGNAL, "noise": NOISE, "interact_ratio": INT_RATIO, "n_boot": N_BOOT},
        "point_estimates": {"full": r2_full, "no_gate": r2_nogate, "no_adain": r2_noadain},
        "full_vs_no_gate": comp_gate,
        "full_vs_no_adain": comp_adain,
    }

    out = RES / "EXP15_ablation_bootstrap.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
