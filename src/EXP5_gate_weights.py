"""EXP-5 · 门控权重可视化

训练 SSP-MFN 后提取门控权重 α 值:
1. 各模态在不同量表维度上的平均门控权重
2. 按民族分组的门控权重差异
3. 按 session 的门控权重变化趋势
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
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


def train_and_extract_gates(data, seed=42):
    """训练模型并提取所有样本的门控权重"""
    N = len(data["y_adj"])
    gkf = GroupKFold(n_splits=5)
    all_gates = []  # (N, 3) 每个样本的 [audio_gate, meta_gate, part_gate]
    all_indices = []

    for fold, (tr, te) in enumerate(gkf.split(data["x_audio"], data["y_adj"], data["groups"])):
        sc_a = StandardScaler().fit(data["x_audio"][tr])
        sc_m = StandardScaler().fit(data["x_meta"][tr])
        sc_p = StandardScaler().fit(data["x_part"][tr])
        sc_y = StandardScaler().fit(data["y_adj"][tr])

        xa_tr = sc_a.transform(data["x_audio"][tr])
        xm_tr = sc_m.transform(data["x_meta"][tr])
        xp_tr = sc_p.transform(data["x_part"][tr])
        y_tr = sc_y.transform(data["y_adj"][tr])

        xa_te = sc_a.transform(data["x_audio"][te])
        xm_te = sc_m.transform(data["x_meta"][te])
        xp_te = sc_p.transform(data["x_part"][te])

        ds_tr = MMDS(xa_tr, xm_tr, xp_tr, y_tr, data["eth_id"][tr])
        dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)

        model = SSPMFN(d_audio=data["audio_dim"], d_meta=data["meta_dim"],
                       d_part=data["part_dim"], d_model=64, n_scales=N_SCALES,
                       n_ethnic=3, use_gate=True, use_adain=True).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.03)
        crit = nn.MSELoss()

        # 训练
        model.train()
        for ep in range(80):
            for xa_b, xm_b, xp_b, y_b, eid_b in dl_tr:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                y_b, eid_b = y_b.to(DEVICE), eid_b.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xa_b, xm_b, xp_b, eid_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # 提取门控权重
        model.eval()
        ds_te = MMDS(xa_te, xm_te, xp_te, np.zeros((len(te), N_SCALES)), data["eth_id"][te])
        dl_te = DataLoader(ds_te, batch_size=len(te), shuffle=False)

        with torch.no_grad():
            for xa_b, xm_b, xp_b, _, eid_b in dl_te:
                xa_b, xm_b, xp_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE)
                eid_b = eid_b.to(DEVICE)
                # 使用 return_weights=True 提取门控
                _, alpha, g = model(xa_b, xm_b, xp_b, eid_b, return_weights=True)
                # alpha: [B, K=6, J=3], g: [B, K=6, J=3]
                # 综合权重 = alpha * g
                weights = (alpha * g).cpu().numpy()  # [B, 6, 3]
                all_gates.append(weights)
                all_indices.append(te)

    all_gates = np.concatenate(all_gates)  # [N, 6, 3]
    all_indices = np.concatenate(all_indices)
    # 按原始顺序排列
    order = np.argsort(all_indices)
    return all_gates[order]


def main():
    print("[EXP-5] 门控权重可视化 (S6 配置)")
    dap = DefensiveProtocol("EXP5_gate_weights")
    data = build_data()
    df = data["df"]
    N = len(df)
    print(f"  N={N}")

    print("  训练并提取门控权重...")
    gates = train_and_extract_gates(data)  # (N, 6, 3): [scales, modalities]
    print(f"  门控权重 shape: {gates.shape}")
    # 对 6 个量表取平均得到模态级权重
    gates_avg = gates.mean(axis=1)  # (N, 3)
    print(f"  均值: audio={gates_avg[:,0].mean():.4f}, meta={gates_avg[:,1].mean():.4f}, part={gates_avg[:,2].mean():.4f}")

    results = {"N": N, "config": "S6"}

    # 1. 逐量表门控权重
    modality_names = ["audio", "meta", "part"]
    results["per_scale_gates"] = {}
    print(f"\n  [1] 逐量表门控权重 (α×g 均值):")
    print(f"    {'量表':<8} {'audio':<10} {'meta':<10} {'part':<10}")
    print(f"    {'-'*38}")
    for k, scale in enumerate(SCALES):
        row = {}
        for j, mn in enumerate(modality_names):
            row[mn] = float(gates[:, k, j].mean())
        results["per_scale_gates"][scale] = row
        print(f"    {scale:<8} {row['audio']:<10.4f} {row['meta']:<10.4f} {row['part']:<10.4f}")

    # 2. 整体门控权重分布
    results["overall_gates"] = {
        "audio": {"mean": float(gates_avg[:,0].mean()), "std": float(gates_avg[:,0].std())},
        "meta": {"mean": float(gates_avg[:,1].mean()), "std": float(gates_avg[:,1].std())},
        "part": {"mean": float(gates_avg[:,2].mean()), "std": float(gates_avg[:,2].std())},
    }
    print(f"\n  [2] 整体门控权重:")
    print(f"    audio: {gates_avg[:,0].mean():.4f} ± {gates_avg[:,0].std():.4f}")
    print(f"    meta:  {gates_avg[:,1].mean():.4f} ± {gates_avg[:,1].std():.4f}")
    print(f"    part:  {gates_avg[:,2].mean():.4f} ± {gates_avg[:,2].std():.4f}")

    # 3. 按民族分组
    results["by_ethnic"] = {}
    print(f"\n  [3] 按民族分组:")
    for eth in ["侗族", "藏族", "蒙古族"]:
        mask = df["ethnic_group"].values == eth
        g = gates_avg[mask]
        results["by_ethnic"][eth] = {
            "n": int(mask.sum()),
            "audio": float(g[:,0].mean()), "meta": float(g[:,1].mean()), "part": float(g[:,2].mean())
        }
        print(f"    {eth} (n={mask.sum()}): audio={g[:,0].mean():.4f} meta={g[:,1].mean():.4f} part={g[:,2].mean():.4f}")

    # 4. 按 session 分组
    results["by_session"] = {}
    print(f"\n  [4] 按 session 分组:")
    for s in sorted(df["session_number"].unique()):
        mask = df["session_number"].values == s
        g = gates_avg[mask]
        results["by_session"][str(int(s))] = {
            "n": int(mask.sum()),
            "audio": float(g[:,0].mean()), "meta": float(g[:,1].mean()), "part": float(g[:,2].mean())
        }
        print(f"    session {int(s)} (n={mask.sum()}): audio={g[:,0].mean():.4f} meta={g[:,1].mean():.4f} part={g[:,2].mean():.4f}")

    # 5. 门控权重与特征的相关
    print(f"\n  [5] 门控权重与特征的相关:")
    age_z = (df["age"].values - df["age"].mean()) / (df["age"].std() + 1e-8)
    mus_z = (df["music_experience_years"].values - df["music_experience_years"].mean()) / (df["music_experience_years"].std() + 1e-8)
    corrs = {}
    for gi, gname in enumerate(modality_names):
        r_age, _ = pearsonr(gates_avg[:, gi], age_z)
        r_mus, _ = pearsonr(gates_avg[:, gi], mus_z)
        corrs[gname] = {"r_age": float(r_age), "r_music_exp": float(r_mus)}
        print(f"    {gname} gate ~ age: r={r_age:.4f}, ~ music_exp: r={r_mus:.4f}")
    results["gate_correlations"] = corrs

    # 6. 民族间门控差异 (ANOVA)
    from scipy.stats import f_oneway
    print(f"\n  [6] 民族间门控差异 (ANOVA):")
    eth_groups = [gates_avg[df["ethnic_group"].values == e] for e in ["侗族", "藏族", "蒙古族"]]
    anova_results = {}
    for gi, gname in enumerate(modality_names):
        F, p = f_oneway(*[g[:, gi] for g in eth_groups])
        anova_results[gname] = {"F": float(F), "p": float(p)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {gname}: F={F:.4f}, p={p:.6f} {sig}")
    results["anova_ethnic"] = anova_results

    # DAP
    gate_stability = [float(gates_avg[:,0].mean()), float(gates_avg[:,1].mean()), float(gates_avg[:,2].mean())]
    dap.check_stability(gate_stability)
    dap_report = dap.generate_report()
    results["DAP"] = dap_report

    out_path = RES / "EXP5_gate_weights.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x,"item") else str(x))
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
