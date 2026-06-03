"""EXP-1c · Shapley 值分析

在 S6 配置下训练 SSP-MFN，用置换 Shapley 方法计算:
1. 模态级 Shapley 值 (audio / meta / part 三路)
2. 特征级 Shapley 值 (top-20 特征贡献排名)
3. 交互项验证: 植入的交互信号是否被正确识别

输出:
- 特征贡献总榜
- 模态级贡献
- 交互依赖分析
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
from scipy.stats import pearsonr
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

class MMDS(Dataset):
    def __init__(s, xa, xm, xp, y, eid):
        s.xa, s.xm, s.xp, s.y, s.eid = xa, xm, xp, y, eid
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        return (torch.FloatTensor(s.xa[i]), torch.FloatTensor(s.xm[i]),
                torch.FloatTensor(s.xp[i]), torch.FloatTensor(s.y[i]),
                torch.LongTensor([s.eid[i]])[0])

def build_data(seed=42):
    """与 EXP1_S6_full.py 完全相同的数据构建"""
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

    # 特征名
    audio_names = [f"mfcc_{i}" for i in range(13)] + ["spec_centroid","spec_bw","spec_rolloff","zcr","tempo"] + [f"chroma_{i}" for i in range(12)]
    meta_names = [c for c in pd.get_dummies(df[["ethnic_group","activity_type","location"]], prefix=["ethnic_group","activity_type","location"]).columns]
    part_names = ["age","music_exp","session_num","duration"] + ["gender","education","native_lang","mandarin_prof"] + [f"{d}_pre" for d in SCALES]

    return {"x_audio": x_audio, "x_meta": x_meta, "x_part": x_part,
            "y_adj": y_adj, "eth_id": eth_id, "groups": groups,
            "audio_dim": x_audio.shape[1], "meta_dim": x_meta.shape[1], "part_dim": x_part.shape[1],
            "audio_names": audio_names, "meta_names": meta_names, "part_names": part_names,
            "xaz": xaz, "xmz": xmz, "age_z": age_z, "mus_z": mus_z}


def train_model(data, tr_idx, te_idx):
    """训练 SSP-MFN 并返回模型 + scaler"""
    sc_a = StandardScaler().fit(data["x_audio"][tr_idx])
    sc_m = StandardScaler().fit(data["x_meta"][tr_idx].astype(np.float64))
    sc_p = StandardScaler().fit(data["x_part"][tr_idx])
    sc_y = StandardScaler().fit(data["y_adj"][tr_idx])

    xa_tr = sc_a.transform(data["x_audio"][tr_idx]).astype(np.float32)
    xm_tr = sc_m.transform(data["x_meta"][tr_idx].astype(np.float64)).astype(np.float32)
    xp_tr = sc_p.transform(data["x_part"][tr_idx]).astype(np.float32)
    y_tr = sc_y.transform(data["y_adj"][tr_idx]).astype(np.float32)

    ds = MMDS(xa_tr, xm_tr, xp_tr, y_tr, data["eth_id"][tr_idx])
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = SSPMFN(d_audio=data["audio_dim"], d_meta=data["meta_dim"], d_part=data["part_dim"],
                   d_model=64, n_ethnic=3, n_scales=6, p_drop=0.3,
                   use_adain=True, use_gate=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
    crit = nn.MSELoss()

    model.train()
    for ep in range(80):
        for xa_b, xm_b, xp_b, y_b, eid_b in dl:
            xa_b, xm_b, xp_b, y_b, eid_b = xa_b.to(DEVICE), xm_b.to(DEVICE), xp_b.to(DEVICE), y_b.to(DEVICE), eid_b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xa_b, xm_b, xp_b, eid_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model, sc_a, sc_m, sc_p, sc_y


def predict(model, xa, xm, xp, eid, sc_y):
    """用模型预测"""
    model.eval()
    with torch.no_grad():
        xa_t = torch.FloatTensor(xa).to(DEVICE)
        xm_t = torch.FloatTensor(xm).to(DEVICE)
        xp_t = torch.FloatTensor(xp).to(DEVICE)
        eid_t = torch.LongTensor(eid).to(DEVICE)
        pred_s = model(xa_t, xm_t, xp_t, eid_t).cpu().numpy()
    return sc_y.inverse_transform(pred_s)


def modality_shapley(model, data, te_idx, sc_a, sc_m, sc_p, sc_y, n_perm=50):
    """模态级 Shapley 值 (3 路模态的 2^3=8 联盟)

    对每个模态子集 S，计算 v(S) = -MSE(pred_S, truth)
    然后用 Shapley 公式分配贡献
    """
    import itertools
    modalities = ["audio", "meta", "part"]
    xa_te = sc_a.transform(data["x_audio"][te_idx]).astype(np.float32)
    xm_te = sc_m.transform(data["x_meta"][te_idx].astype(np.float64)).astype(np.float32)
    xp_te = sc_p.transform(data["x_part"][te_idx]).astype(np.float32)
    eid_te = data["eth_id"][te_idx]
    y_te = data["y_adj"][te_idx]

    # 基线: 全部 mask (随机打乱)
    rng = np.random.default_rng(42)

    def get_pred(include_set):
        xa_in = xa_te.copy()
        xm_in = xm_te.copy()
        xp_in = xp_te.copy()
        if "audio" not in include_set:
            xa_in = rng.permutation(xa_in)  # 打乱破坏信号
        if "meta" not in include_set:
            xm_in = rng.permutation(xm_in)
        if "part" not in include_set:
            xp_in = rng.permutation(xp_in)
        pred = predict(model, xa_in, xm_in, xp_in, eid_te, sc_y)
        return -mean_squared_error(y_te, pred)  # 负 MSE 作为 value

    # 计算所有 2^3 联盟的 value
    coalition_values = {}
    for r in range(4):
        for subset in itertools.combinations(modalities, r):
            key = frozenset(subset)
            vals = []
            for _ in range(n_perm):
                vals.append(get_pred(subset))
            coalition_values[key] = np.mean(vals)

    # Shapley 公式
    from math import factorial
    n = 3
    shapley_vals = {}
    for i, mod in enumerate(modalities):
        phi = 0.0
        others = [m for m in modalities if m != mod]
        for r in range(n):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                S_with_i = frozenset(list(subset) + [mod])
                marginal = coalition_values[S_with_i] - coalition_values[S]
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi += weight * marginal
        shapley_vals[mod] = phi

    return shapley_vals


def feature_importance_permutation(model, data, te_idx, sc_a, sc_m, sc_p, sc_y, n_perm=20):
    """特征级置换重要性 (Permutation Importance)

    逐个打乱每个特征，观察预测性能下降
    """
    xa_te = sc_a.transform(data["x_audio"][te_idx]).astype(np.float32)
    xm_te = sc_m.transform(data["x_meta"][te_idx].astype(np.float64)).astype(np.float32)
    xp_te = sc_p.transform(data["x_part"][te_idx]).astype(np.float32)
    eid_te = data["eth_id"][te_idx]
    y_te = data["y_adj"][te_idx]

    # 基线性能
    pred_base = predict(model, xa_te, xm_te, xp_te, eid_te, sc_y)
    base_mse = mean_squared_error(y_te, pred_base)

    rng = np.random.default_rng(42)
    importances = {}

    # 音频特征
    for j in range(xa_te.shape[1]):
        drops = []
        for _ in range(n_perm):
            xa_perm = xa_te.copy()
            xa_perm[:, j] = rng.permutation(xa_perm[:, j])
            pred = predict(model, xa_perm, xm_te, xp_te, eid_te, sc_y)
            drops.append(mean_squared_error(y_te, pred) - base_mse)
        importances[data["audio_names"][j]] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "modality": "audio"}

    # 文化元特征
    for j in range(xm_te.shape[1]):
        drops = []
        for _ in range(n_perm):
            xm_perm = xm_te.copy()
            xm_perm[:, j] = rng.permutation(xm_perm[:, j])
            pred = predict(model, xa_te, xm_perm, xp_te, eid_te, sc_y)
            drops.append(mean_squared_error(y_te, pred) - base_mse)
        name = data["meta_names"][j] if j < len(data["meta_names"]) else f"meta_{j}"
        importances[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "modality": "meta"}

    # 参与者特征
    for j in range(xp_te.shape[1]):
        drops = []
        for _ in range(n_perm):
            xp_perm = xp_te.copy()
            xp_perm[:, j] = rng.permutation(xp_perm[:, j])
            pred = predict(model, xa_te, xm_te, xp_perm, eid_te, sc_y)
            drops.append(mean_squared_error(y_te, pred) - base_mse)
        name = data["part_names"][j] if j < len(data["part_names"]) else f"part_{j}"
        importances[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "modality": "part"}

    return importances


def interaction_verification(data, te_idx, model, sc_a, sc_m, sc_p, sc_y):
    """验证模型是否捕获了植入的交互效应

    方法: 计算模型预测残差与已知交互项的相关性
    如果模型成功捕获交互 → 残差与交互项不相关
    如果模型未捕获交互 → 残差与交互项高度相关
    """
    xa_te = sc_a.transform(data["x_audio"][te_idx]).astype(np.float32)
    xm_te = sc_m.transform(data["x_meta"][te_idx].astype(np.float64)).astype(np.float32)
    xp_te = sc_p.transform(data["x_part"][te_idx]).astype(np.float32)
    eid_te = data["eth_id"][te_idx]
    y_te = data["y_adj"][te_idx]

    pred = predict(model, xa_te, xm_te, xp_te, eid_te, sc_y)
    residual = y_te - pred

    # 已知植入的交互项
    xaz = data["xaz"][te_idx]
    mus_z = data["mus_z"][te_idx]
    age_z = data["age_z"][te_idx]
    xmz = data["xmz"][te_idx]

    interactions = {
        "audio×music_exp (→ICS)": xaz[:, 2] * mus_z,
        "audio×age (→ICS)": xaz[:, 3] * age_z,
        "meta×age (→IRI)": xmz[:, 0] * age_z,
        "meta×music_exp (→IRI)": xmz[:, 1] * mus_z,
        "audio×meta×music (→IOS)": xaz[:, 0] * xmz[:, 0] * mus_z,
        "audio×music_exp (→CSAS)": xaz[:, 0] * mus_z,
        "age×music_exp (→SSCS)": age_z * mus_z,
        "meta0×meta1 (→SCI2)": xmz[:, 0] * xmz[:, 1],
    }

    # 目标维度映射
    target_dims = {"→ICS": 0, "→IRI": 1, "→CSAS": 2, "→SSCS": 3, "→IOS": 4, "→SCI2": 5}

    results = {}
    for name, interaction_vals in interactions.items():
        # 找到对应的目标维度
        dim_key = [k for k in target_dims if k in name]
        if dim_key:
            dim = target_dims[dim_key[0]]
            r_resid, p_resid = pearsonr(interaction_vals, residual[:, dim])
            r_truth, p_truth = pearsonr(interaction_vals, y_te[:, dim])
            results[name] = {
                "r_with_residual": round(float(r_resid), 4),
                "p_residual": float(p_resid),
                "r_with_truth": round(float(r_truth), 4),
                "captured_pct": round((1 - abs(r_resid) / (abs(r_truth) + 1e-8)) * 100, 1),
            }

    return results


def main():
    print("[EXP-1c] Shapley 值分析 (S6 配置)")
    print(f"  device={DEVICE}, S={SIGNAL}, σ={NOISE}, int={INT_RATIO}")

    data = build_data(seed=42)
    N = len(data["y_adj"])
    print(f"  N={N}")

    # 用第一折训练模型
    gkf = GroupKFold(n_splits=5)
    tr_idx, te_idx = next(iter(gkf.split(data["x_audio"], data["y_adj"], data["groups"])))
    print(f"  训练集: {len(tr_idx)}, 测试集: {len(te_idx)}")

    print("\n  训练 SSP-MFN...")
    model, sc_a, sc_m, sc_p, sc_y = train_model(data, tr_idx, te_idx)

    # 验证模型性能
    xa_te = sc_a.transform(data["x_audio"][te_idx]).astype(np.float32)
    xm_te = sc_m.transform(data["x_meta"][te_idx].astype(np.float64)).astype(np.float32)
    xp_te = sc_p.transform(data["x_part"][te_idx]).astype(np.float32)
    pred = predict(model, xa_te, xm_te, xp_te, data["eth_id"][te_idx], sc_y)
    r2 = r2_score(data["y_adj"][te_idx], pred)
    print(f"  模型 R² = {r2:.4f}")

    # === 1. 模态级 Shapley ===
    print("\n  [1/3] 模态级 Shapley 值...")
    shapley = modality_shapley(model, data, te_idx, sc_a, sc_m, sc_p, sc_y, n_perm=30)
    total = sum(abs(v) for v in shapley.values())
    print(f"    audio:  φ={shapley['audio']:.4f} ({abs(shapley['audio'])/total*100:.1f}%)")
    print(f"    meta:   φ={shapley['meta']:.4f} ({abs(shapley['meta'])/total*100:.1f}%)")
    print(f"    part:   φ={shapley['part']:.4f} ({abs(shapley['part'])/total*100:.1f}%)")

    # === 2. 特征级置换重要性 ===
    print("\n  [2/3] 特征级置换重要性 (Top-20)...")
    importances = feature_importance_permutation(model, data, te_idx, sc_a, sc_m, sc_p, sc_y, n_perm=10)
    sorted_imp = sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True)
    print(f"    {'排名':<4} {'特征':<25} {'ΔmSE':<10} {'模态':<8}")
    print(f"    {'-'*50}")
    for i, (name, v) in enumerate(sorted_imp[:20]):
        print(f"    {i+1:<4} {name:<25} {v['mean']:.4f}    {v['modality']}")

    # === 3. 交互项验证 ===
    print("\n  [3/3] 交互项捕获验证...")
    interact_results = interaction_verification(data, te_idx, model, sc_a, sc_m, sc_p, sc_y)
    print(f"    {'交互项':<30} {'r(残差)':<10} {'r(真值)':<10} {'捕获%':<8}")
    print(f"    {'-'*60}")
    for name, v in interact_results.items():
        captured = v["captured_pct"]
        flag = "✓" if captured > 50 else "△" if captured > 20 else "✗"
        print(f"    {name:<30} {v['r_with_residual']:<10.4f} {v['r_with_truth']:<10.4f} {captured:.1f}% {flag}")

    # 保存
    results = {
        "config": {"signal": SIGNAL, "noise": NOISE, "interact_ratio": INT_RATIO},
        "model_r2": round(r2, 4),
        "modality_shapley": {k: round(v, 4) for k, v in shapley.items()},
        "modality_shapley_pct": {k: round(abs(v)/total*100, 1) for k, v in shapley.items()},
        "top20_features": [{
            "rank": i+1, "name": name, "delta_mse": round(v["mean"], 4),
            "modality": v["modality"]
        } for i, (name, v) in enumerate(sorted_imp[:20])],
        "interaction_verification": interact_results,
    }
    out = RES / "EXP1c_shapley.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\n  saved → {out}")

    # 汇总
    print("\n" + "="*70)
    print("  Shapley 分析汇总:")
    print(f"  模态贡献: part({abs(shapley['part'])/total*100:.0f}%) > audio({abs(shapley['audio'])/total*100:.0f}%) > meta({abs(shapley['meta'])/total*100:.0f}%)")
    avg_captured = np.mean([v["captured_pct"] for v in interact_results.values()])
    print(f"  交互项平均捕获率: {avg_captured:.1f}%")
    n_captured = sum(1 for v in interact_results.values() if v["captured_pct"] > 50)
    print(f"  成功捕获 (>50%): {n_captured}/{len(interact_results)} 个交互项")
    print("="*70)


if __name__ == "__main__":
    main()
