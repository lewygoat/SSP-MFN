"""论文可视化图表生成 (DPI=600)

Fig 1: 模型对比柱状图 (R² + RMSE)
Fig 2: 消融矩阵热力图 (2³ 组合 × 6 量表)
Fig 3: 门控权重热力图 (6 量表 × 3 模态)
Fig 4: 门控权重按民族分组
Fig 5: Bootstrap CI 森林图
Fig 6: 模态边际贡献
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
FIG = ROOT / "实验/figures"; FIG.mkdir(parents=True, exist_ok=True)
DPI = 600

# 加载数据
with open(RES/"EXP1_S6_full.json") as f: exp1 = json.load(f)
with open(RES/"EXP2_ablation_matrix.json") as f: exp2 = json.load(f)
with open(RES/"EXP5_gate_weights.json") as f: exp5 = json.load(f)
with open(RES/"EXP7_bootstrap_ci.json") as f: exp7 = json.load(f)

SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]

# ============================================================
# Fig 1: 模型对比柱状图
# ============================================================
print("  Fig 1: 模型对比...")
models_order = ["B1_pre_only","B2_Ridge","B3_Lasso","B4_ElasticNet",
                "B5_RF","B6_XGBoost","B7_SVR","B8_KNN",
                "M1_SSP_MFN_full","M2_SSP_MFN_no_gate",
                "M3_SSP_MFN_no_adain","M4_SSP_MFN_plain"]
labels = ["Pre-only","Ridge","Lasso","ElasticNet","RF","XGBoost","SVR","KNN",
          "SSP-MFN\n(full)","SSP-MFN\n(no gate)","SSP-MFN\n(no AdaIN)","SSP-MFN\n(plain)"]
r2_vals = [exp1["N850"][m]["_mean"]["r2"] for m in models_order]
rmse_vals = [exp1["N850"][m]["_mean"]["rmse"] for m in models_order]

colors = ['#95a5a6']*8 + ['#e74c3c','#e67e22','#f39c12','#f1c40f']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

bars1 = ax1.bar(range(len(models_order)), r2_vals, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('(a) Model Comparison: R² (higher is better)', fontsize=13, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax1.set_ylim(-0.05, 0.15)
for i, v in enumerate(r2_vals):
    ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

bars2 = ax2.bar(range(len(models_order)), rmse_vals, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_title('(b) Model Comparison: RMSE (lower is better)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(models_order)))
ax2.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
for i, v in enumerate(rmse_vals):
    ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig(FIG/"fig1_model_comparison.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig1_model_comparison.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 2: 消融矩阵热力图
# ============================================================
print("  Fig 2: 消融矩阵热力图...")
combos = ["__P","_M_","_MP","A__","A_P","AM_","AMP"]
combo_labels = ["Part only","Meta only","Meta+Part","Audio only",
                "Audio+Part","Audio+Meta","All (AMP)"]
matrix = np.zeros((len(combos), len(SCALES)))
for i, c in enumerate(combos):
    for j, s in enumerate(SCALES):
        matrix[i, j] = exp2[c][s]["r2"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.25)
ax.set_xticks(range(len(SCALES)))
ax.set_xticklabels(SCALES, fontsize=11)
ax.set_yticks(range(len(combos)))
ax.set_yticklabels(combo_labels, fontsize=10)
ax.set_xlabel('Social Skill Scale', fontsize=12)
ax.set_ylabel('Modality Combination', fontsize=12)
ax.set_title('Ablation Matrix: R² per Scale × Modality Combination', fontsize=13, fontweight='bold')

for i in range(len(combos)):
    for j in range(len(SCALES)):
        v = matrix[i, j]
        color = 'white' if abs(v) > 0.15 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('R²', fontsize=11)
plt.tight_layout()
plt.savefig(FIG/"fig2_ablation_heatmap.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig2_ablation_heatmap.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 3: 门控权重热力图 (6 量表 × 3 模态)
# ============================================================
print("  Fig 3: 门控权重热力图...")
gate_matrix = np.zeros((len(SCALES), 3))
modality_names = ["Audio", "Meta", "Part"]
for i, s in enumerate(SCALES):
    for j, m in enumerate(["audio","meta","part"]):
        gate_matrix[i, j] = exp5["per_scale_gates"][s][m]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(gate_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.35)
ax.set_xticks(range(3))
ax.set_xticklabels(modality_names, fontsize=12)
ax.set_yticks(range(len(SCALES)))
ax.set_yticklabels(SCALES, fontsize=11)
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('Scale', fontsize=12)
ax.set_title('Gating Weights (α × g) per Scale × Modality', fontsize=13, fontweight='bold')

for i in range(len(SCALES)):
    for j in range(3):
        v = gate_matrix[i, j]
        color = 'white' if v > 0.25 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=11, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Weight', fontsize=11)
plt.tight_layout()
plt.savefig(FIG/"fig3_gate_heatmap.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig3_gate_heatmap.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 4: 门控权重按民族分组 (分组柱状图)
# ============================================================
print("  Fig 4: 门控权重按民族...")
ethnic_names = ["Dong (侗族)", "Tibetan (藏族)", "Mongolian (蒙古族)"]
ethnic_keys = ["侗族", "藏族", "蒙古族"]
x = np.arange(3)
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, (ek, en) in enumerate(zip(ethnic_keys, ethnic_names)):
    vals = [exp5["by_ethnic"][ek]["audio"], exp5["by_ethnic"][ek]["meta"], exp5["by_ethnic"][ek]["part"]]
    ax.bar(x + i*width, vals, width, label=en, edgecolor='white', linewidth=0.5)

ax.set_xticks(x + width)
ax.set_xticklabels(modality_names, fontsize=12)
ax.set_ylabel('Gate Weight (α × g)', fontsize=12)
ax.set_title('Gating Weights by Ethnic Group', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 0.32)

# 添加 ANOVA 显著性标记
for j, m in enumerate(["audio","meta","part"]):
    p = exp5["anova_ethnic"][m]["p"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ymax = max(exp5["by_ethnic"][ek][m] for ek in ethnic_keys)
    ax.text(j + width, ymax + 0.015, sig, ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG/"fig4_gate_by_ethnic.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig4_gate_by_ethnic.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 5: Bootstrap CI 森林图
# ============================================================
print("  Fig 5: Bootstrap CI 森林图...")
comparisons = exp7["comparisons"]
comp_names = list(comparisons.keys())
deltas = [comparisons[c]["mean_diff"] for c in comp_names]
ci_lo = [comparisons[c]["ci_95_lo"] for c in comp_names]
ci_hi = [comparisons[c]["ci_95_hi"] for c in comp_names]
pvals = [comparisons[c]["p_value"] for c in comp_names]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = range(len(comp_names))
xerr_lo = [d - lo for d, lo in zip(deltas, ci_lo)]
xerr_hi = [hi - d for d, hi in zip(deltas, ci_hi)]

colors_ci = []
for p in pvals:
    if p < 0.01: colors_ci.append('#27ae60')
    elif p < 0.05: colors_ci.append('#2ecc71')
    else: colors_ci.append('#95a5a6')

ax.barh(y_pos, deltas, xerr=[xerr_lo, xerr_hi], color=colors_ci,
        edgecolor='white', linewidth=0.5, capsize=4, height=0.6)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"vs {n}" for n in comp_names], fontsize=11)
ax.set_xlabel('ΔR² (SSP-MFN − Baseline)', fontsize=12)
ax.set_title('Bootstrap 95% CI: SSP-MFN vs Baselines', fontsize=13, fontweight='bold')

for i, (d, p) in enumerate(zip(deltas, pvals)):
    sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(max(ci_hi) + 0.005, i, f'p={p:.3f} {sig}', va='center', fontsize=9)

ax.set_xlim(-0.04, max(ci_hi) + 0.06)
plt.tight_layout()
plt.savefig(FIG/"fig5_bootstrap_ci.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig5_bootstrap_ci.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 6: 消融组件贡献 (AdaIN vs Gate vs Both)
# ============================================================
print("  Fig 6: 组件消融对比...")
ablation_models = ["M1_SSP_MFN_full","M2_SSP_MFN_no_gate","M3_SSP_MFN_no_adain","M4_SSP_MFN_plain"]
ablation_labels = ["Full\n(Gate+AdaIN)","No Gate\n(AdaIN only)","No AdaIN\n(Gate only)","Plain\n(Neither)"]
ablation_r2 = [exp1["N850"][m]["_mean"]["r2"] for m in ablation_models]

fig, ax = plt.subplots(figsize=(7, 5))
colors_abl = ['#e74c3c','#e67e22','#f39c12','#95a5a6']
bars = ax.bar(range(4), ablation_r2, color=colors_abl, edgecolor='white', linewidth=0.5, width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(ablation_labels, fontsize=11)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Component Ablation: Contribution of Gate and AdaIN', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.15)

for i, v in enumerate(ablation_r2):
    ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# 添加贡献标注
ax.annotate('', xy=(0, ablation_r2[0]), xytext=(2, ablation_r2[2]),
            arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.5))
ax.text(1, (ablation_r2[0]+ablation_r2[2])/2 + 0.005, 
        f'AdaIN: +{ablation_r2[0]-ablation_r2[2]:.4f}', 
        ha='center', fontsize=9, color='#2c3e50', fontweight='bold')

ax.annotate('', xy=(0, ablation_r2[0]-0.002), xytext=(1, ablation_r2[1]-0.002),
            arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.5))
ax.text(0.5, min(ablation_r2[0], ablation_r2[1]) - 0.012,
        f'Gate: +{ablation_r2[0]-ablation_r2[1]:.4f}',
        ha='center', fontsize=9, color='#8e44ad', fontweight='bold')

plt.tight_layout()
plt.savefig(FIG/"fig6_component_ablation.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig6_component_ablation.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 7: 门控权重按 session 变化趋势
# ============================================================
print("  Fig 7: 门控权重趋势...")
sessions = sorted(exp5["by_session"].keys(), key=int)
audio_trend = [exp5["by_session"][s]["audio"] for s in sessions]
meta_trend = [exp5["by_session"][s]["meta"] for s in sessions]
part_trend = [exp5["by_session"][s]["part"] for s in sessions]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = [int(s) for s in sessions]
ax.plot(x, audio_trend, 'o-', color='#3498db', linewidth=2, markersize=8, label='Audio')
ax.plot(x, meta_trend, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Meta')
ax.plot(x, part_trend, '^-', color='#27ae60', linewidth=2, markersize=8, label='Part')
ax.set_xlabel('Session Number', fontsize=12)
ax.set_ylabel('Gate Weight (α × g)', fontsize=12)
ax.set_title('Gating Weight Dynamics Across Sessions', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(x)
ax.set_ylim(0, 0.30)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG/"fig7_gate_session_trend.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"fig7_gate_session_trend.pdf", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"\n  所有图表已保存至: {FIG}/")
print(f"  共 7 张图 (PNG + PDF), DPI={DPI}")
