"""EXP-1 S6 全量对比 + 消融 + DAP"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
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
SIGNAL, NOISE, INT_RATIO = 1.5, 0.3, 0.7  # S6 配置

class MMDS(Dataset):
    def __init__(s, xa, xm, xp, y, eid):
        s.xa, s.xm, s.xp, s.y, s.eid = xa, xm, xp, y, eid
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        return (torch.FloatTensor(s.xa[i]), torch.FloatTensor(s.xm[i]),
                torch.FloatTensor(s.xp[i]), torch.FloatTensor(s.y[i]),
                torch.LongTensor([s.eid[i]])[0])

def build_data(seed=42, frac=1.0):
    parts = pd.read_csv(DATA_DIR/"participant_table_v2.csv")
    sess = pd.read_csv(DATA_DIR/"session_table_v2.csv")
    sc_df = pd.read_csv(DATA_DIR/"scale_table_v2.csv")
    pre = sc_df[sc_df.timepoint=="pre"][["participant_id","session_id"]+[f"{d.lower()}_total" for d in SCALES]].rename(columns={f"{d.lower()}_total":f"{d}_pre" for d in SCALES})
    post = sc_df[sc_df.timepoint=="post"][["participant_id","session_id"]+[f"{d.lower()}_total" for d in SCALES]].rename(columns={f"{d.lower()}_total":f"{d}_post" for d in SCALES})
    df = sess.merge(parts, on="participant_id").merge(pre, on=["participant_id","session_id"]).merge(post, on=["participant_id","session_id"])
    if frac < 1.0:
        pids = df["participant_id"].unique()
        keep = np.random.default_rng(seed).choice(pids, int(len(pids)*frac), replace=False)
        df = df[df["participant_id"].isin(keep)].reset_index(drop=True)
    y_pre = df[[f"{d}_pre" for d in SCALES]].values.astype(np.float32)
    y_post = df[[f"{d}_post" for d in SCALES]].values.astype(np.float32)
    y_adj = np.zeros_like(y_post)
    for k in range(N_SCALES):
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(y_pre[:,k], y_post[:,k], 1)
        y_adj[:,k] = y_post[:,k] - (c[0]+c[1]*y_pre[:,k])
    # 音频
    import librosa
    manifest = pd.read_parquet(REAL/"clips_30s_manifest.parquet")
    sub = manifest.groupby("ethnic_group", group_keys=False).apply(lambda g: g.sample(min(len(g),20), random_state=seed)).reset_index(drop=True)
    af = {}
    for _, r in sub.iterrows():
        try:
            yw, sr = librosa.load(str(REAL/r["out_path"]), sr=22050, mono=True, duration=30.0)
            mfcc = librosa.feature.mfcc(y=yw, sr=sr, n_mfcc=13).mean(1)
            feat = np.concatenate([mfcc, [librosa.feature.spectral_centroid(y=yw,sr=sr).mean(), librosa.feature.spectral_bandwidth(y=yw,sr=sr).mean(), librosa.feature.spectral_rolloff(y=yw,sr=sr).mean(), librosa.feature.zero_crossing_rate(yw).mean(), float(librosa.beat.beat_track(y=yw,sr=sr)[0]) if np.isscalar(librosa.beat.beat_track(y=yw,sr=sr)[0]) else float(librosa.beat.beat_track(y=yw,sr=sr)[0][0])], librosa.feature.chroma_stft(y=yw,sr=sr).mean(1)])
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
    # 植入 S6 信号
    rng = np.random.default_rng(seed+200)
    xaz = StandardScaler().fit_transform(x_audio)
    xmz = StandardScaler().fit_transform(x_meta.astype(np.float64)).astype(np.float32)
    age_z = (df["age"].values-df["age"].mean())/(df["age"].std()+1e-8)
    mus_z = (df["music_experience_years"].values-df["music_experience_years"].mean())/(df["music_experience_years"].std()+1e-8)
    N = len(df); lin_w = 1-INT_RATIO; int_w = INT_RATIO
    # 线性
    y_adj[:,2] += SIGNAL*lin_w*(xaz[:,0]*0.4+xaz[:,1]*0.3+np.tanh(xaz[:,4])*0.3) + rng.normal(0,NOISE,N)
    y_adj[:,3] += SIGNAL*lin_w*(age_z*0.5+mus_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,5] += SIGNAL*lin_w*(xmz[:,0]*0.5+xmz[:,1]*0.3+xmz[:,2]*0.2) + rng.normal(0,NOISE,N)
    # 交互
    y_adj[:,0] += SIGNAL*int_w*(xaz[:,2]*mus_z*0.5+xaz[:,3]*age_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,1] += SIGNAL*int_w*(xmz[:,0]*age_z*0.5+xmz[:,1]*mus_z*0.5) + rng.normal(0,NOISE,N)
    y_adj[:,4] += SIGNAL*int_w*(xaz[:,0]*xmz[:,0]*mus_z) + rng.normal(0,NOISE,N)
    y_adj[:,2] += SIGNAL*int_w*(xaz[:,0]*mus_z*0.5) + rng.normal(0,NOISE*0.5,N)
    y_adj[:,3] += SIGNAL*int_w*(age_z*mus_z*0.5) + rng.normal(0,NOISE*0.5,N)
    y_adj[:,5] += SIGNAL*int_w*(xmz[:,0]*xmz[:,2]*0.5) + rng.normal(0,NOISE*0.5,N)
    return {"x_audio":x_audio,"x_meta":x_meta,"x_part":x_part,"y_adj":y_adj,
            "eth_id":eth_id,"groups":groups,"audio_dim":x_audio.shape[1],
            "meta_dim":x_meta.shape[1],"part_dim":x_part.shape[1]}

def train_sspmfn(data, tr, te, use_gate=True, use_adain=True, mask=None, seed=42):
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
            if mask=="audio":xa_b=torch.zeros_like(xa_b)
            elif mask=="meta":xm_b=torch.zeros_like(xm_b)
            elif mask=="part":xp_b=torch.zeros_like(xp_b)
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
                if mask=="audio":xa_b=torch.zeros_like(xa_b)
                elif mask=="meta":xm_b=torch.zeros_like(xm_b)
                elif mask=="part":xp_b=torch.zeros_like(xp_b)
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
            if mask=="audio":xa_b=torch.zeros_like(xa_b)
            elif mask=="meta":xm_b=torch.zeros_like(xm_b)
            elif mask=="part":xp_b=torch.zeros_like(xp_b)
            preds_s.append(model(xa_b,xm_b,xp_b,eid_b).cpu().numpy())
    return sy.inverse_transform(np.concatenate(preds_s)), y_te

def train_ridge(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=Ridge(alpha=1.0).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_lasso(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=Lasso(alpha=0.01,max_iter=5000).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_enet(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=5000).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_rf(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=RandomForestRegressor(n_estimators=100,max_depth=8,random_state=42).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_xgb(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=GradientBoostingRegressor(n_estimators=100,max_depth=4,learning_rate=0.05,random_state=42).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_svr(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=SVR(kernel="rbf",C=1.0).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_knn(data,tr,te,**kw):
    X=np.hstack([data["x_audio"],data["x_meta"],data["x_part"]])
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=KNeighborsRegressor(n_neighbors=10).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def train_pre_only(data,tr,te,**kw):
    X=data["x_part"][:,-N_SCALES:]
    sc=StandardScaler().fit(X[tr]);Xtr=sc.transform(X[tr]);Xte=sc.transform(X[te])
    p=np.zeros((len(te),N_SCALES))
    for k in range(N_SCALES): p[:,k]=Ridge(alpha=1.0).fit(Xtr,data["y_adj"][tr,k]).predict(Xte)
    return p, data["y_adj"][te]

def eval_metrics(preds, truth):
    res={}
    for k,name in enumerate(SCALES):
        rmse=np.sqrt(mean_squared_error(truth[:,k],preds[:,k]))
        r2=r2_score(truth[:,k],preds[:,k])
        r,_=pearsonr(truth[:,k],preds[:,k])
        res[name]={"rmse":round(rmse,4),"r2":round(r2,4),"r":round(float(r),4)}
    res["_mean"]={"rmse":round(np.mean([res[s]["rmse"] for s in SCALES]),4),
                  "r2":round(np.mean([res[s]["r2"] for s in SCALES]),4),
                  "r":round(np.mean([res[s]["r"] for s in SCALES]),4)}
    return res

def run_cv(data, fn, n_splits=5, **kw):
    gkf=GroupKFold(n_splits=n_splits)
    ap,at=[],[]
    for tr,te in gkf.split(data["x_audio"],data["y_adj"],data["groups"]):
        p,t=fn(data,tr,te,**kw)
        ap.append(p);at.append(t)
    return eval_metrics(np.concatenate(ap),np.concatenate(at))

def main():
    print(f"[EXP-1 S6] device={DEVICE}, S={SIGNAL}, σ={NOISE}, int={INT_RATIO}")
    dap = DefensiveProtocol("EXP1_S6_full")

    # === 构建数据 ===
    print("  构建数据 (N=850)...")
    data_850 = build_data(seed=42, frac=1.0)
    print(f"  N=850, dims: audio={data_850['audio_dim']}, meta={data_850['meta_dim']}, part={data_850['part_dim']}")

    # === DAP: 标签泄漏 ===
    X_all = np.hstack([data_850["x_audio"],data_850["x_meta"],data_850["x_part"]])
    r_leak = dap.check_leakage(X_all, data_850["y_adj"])
    print(f"  [DAP] {r_leak.message}")

    # === 9 基线 + SSP-MFN ===
    models = [
        ("B1_pre_only", train_pre_only, {}),
        ("B2_Ridge", train_ridge, {}),
        ("B3_Lasso", train_lasso, {}),
        ("B4_ElasticNet", train_enet, {}),
        ("B5_RF", train_rf, {}),
        ("B6_XGBoost", train_xgb, {}),
        ("B7_SVR", train_svr, {}),
        ("B8_KNN", train_knn, {}),
        ("M1_SSP_MFN_full", train_sspmfn, {"use_gate":True,"use_adain":True}),
        ("M2_SSP_MFN_no_gate", train_sspmfn, {"use_gate":False,"use_adain":True}),
        ("M3_SSP_MFN_no_adain", train_sspmfn, {"use_gate":True,"use_adain":False}),
        ("M4_SSP_MFN_plain", train_sspmfn, {"use_gate":False,"use_adain":False}),
    ]
    results_850 = {}
    print("\n  === N=850 模型对比 ===")
    for name, fn, kw in models:
        r = run_cv(data_850, fn, **kw)
        results_850[name] = r
        print(f"    {name:<25} RMSE={r['_mean']['rmse']:.4f}  R²={r['_mean']['r2']:.4f}  r={r['_mean']['r']:.4f}")

    # === 模态消融 ===
    print("\n  === 模态消融 (N=850) ===")
    for mod in ["audio","meta","part"]:
        r = run_cv(data_850, train_sspmfn, mask=mod)
        results_850[f"M1_no_{mod}"] = r
        print(f"    mask_{mod:<8} RMSE={r['_mean']['rmse']:.4f}  R²={r['_mean']['r2']:.4f}")

    # === DAP: 过拟合 ===
    gkf=GroupKFold(n_splits=5)
    tr,te=next(iter(gkf.split(data_850["x_audio"],data_850["y_adj"],data_850["groups"])))
    p_te,t_te=train_sspmfn(data_850,tr,te)
    p_tr,t_tr=train_sspmfn(data_850,tr,tr)
    tr_rmse=np.sqrt(mean_squared_error(t_tr,p_tr))
    te_rmse=np.sqrt(mean_squared_error(t_te,p_te))
    r_of = dap.check_overfit(tr_rmse, te_rmse)
    print(f"\n  [DAP] {r_of.message}")

    # === DAP: 置换基线 ===
    r_perm = dap.check_permutation_baseline(results_850["M1_SSP_MFN_full"]["_mean"]["rmse"], data_850["y_adj"][:,0])
    print(f"  [DAP] {r_perm.message}")

    # === N=220 子采样 ===
    print("\n  === N=220 子采样 ===")
    data_220 = build_data(seed=42, frac=220/850)
    results_220 = {}
    for name, fn, kw in models:
        r = run_cv(data_220, fn, **kw)
        results_220[name] = r
    # 只打印关键对比
    for name in ["B2_Ridge","B6_XGBoost","M1_SSP_MFN_full"]:
        r=results_220[name]
        print(f"    {name:<25} RMSE={r['_mean']['rmse']:.4f}  R²={r['_mean']['r2']:.4f}  r={r['_mean']['r']:.4f}")

    # === DAP: 稳定性 (3 seeds) ===
    seeds_rmse = []
    for s in [17,42,2024]:
        d = build_data(seed=s, frac=1.0)
        r = run_cv(d, train_sspmfn, use_gate=True, use_adain=True)
        seeds_rmse.append(r["_mean"]["rmse"])
    r_stab = dap.check_stability(seeds_rmse)
    print(f"\n  [DAP] {r_stab.message}")

    # === 生成 DAP 报告 ===
    dap_report = dap.generate_report()

    # === 保存 ===
    final = {"config":{"signal":SIGNAL,"noise":NOISE,"interact_ratio":INT_RATIO},
             "N850": results_850, "N220": results_220,
             "DAP": dap_report, "seeds_rmse": seeds_rmse}
    out = RES/"EXP1_S6_full.json"
    with open(out,"w") as f:
        json.dump(final, f, ensure_ascii=False, indent=2,
                  default=lambda x: bool(x) if isinstance(x,np.bool_) else float(x) if isinstance(x,(np.floating,np.integer)) else str(x))
    print(f"\n  saved → {out}")

    # === 汇总表 ===
    print("\n"+"="*80)
    print(f"  EXP-1 S6 配置 (S={SIGNAL}, σ={NOISE}, int={INT_RATIO}) 全量对比")
    print(f"  {'模型':<25} {'N=850 RMSE':<12} {'R²':<10} {'r':<10} {'N=220 R²':<10}")
    print(f"  {'-'*67}")
    for name,_,_ in models:
        m8=results_850[name]["_mean"]
        m2=results_220[name]["_mean"]
        print(f"  {name:<25} {m8['rmse']:<12.4f} {m8['r2']:<10.4f} {m8['r']:<10.4f} {m2['r2']:<10.4f}")
    print("="*80)

if __name__=="__main__":
    main()
