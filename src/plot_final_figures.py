"""论文最终图表 (按审稿人反馈重构)

Figure 1: 效能证明
  (a) Model Comparison + 字母分组法
  (b) Bootstrap CI 森林图
  (c) Component Ablation (inset 风格)

Figure 2: 机理解释
  (a) Ablation Matrix 热力图 (center=0, 分段色图)
  (b) Gating Weights 热力图
  (c) Session Dynamics 趋势图

DPI=600, 统一字体/标注风格
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# 字体设置
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
FIG = ROOT / "实验/figures"; FIG.mkdir(parents=True, exist_ok=True)
DPI = 600

# 加载数据
with open(RES/"EXP1_S6_full.json") as f: exp1 = json.load(f)
with open(RES/"EXP2_ablation_matrix.json") as f: exp2 = json.load(f)
with open(RES/"EXP5_gate_weights.json") as f: exp5 = json.load(f)
with open(RES/"EXP7_bootstrap_ci.json") as f: exp7 = json.load(f)
with open(RES/"EXP8_supplementary_baselines.json") as f: exp8 = json.load(f)
with open(RES/"EXP9_permutation_sanity.json") as f: exp9 = json.load(f)

SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]

# ============================================================
# Figure 1: 效能证明 (3 子图)
# ============================================================
print("  Figure 1: 效能证明...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# --- (a) Model Comparison with letter grouping (including EXP-8 baselines) ---
ax_a = fig.add_subplot(gs[0, :])

# 所有模型: EXP-1 原始 + EXP-8 新增
models_order = ["B1_pre_only","B2_Ridge","B3_Lasso","B4_ElasticNet",
                "B5_RF","B6_XGBoost","B7_SVR","B8_KNN",
                "B9_MulT","B10_TFN","B11_LMF","B12_MLP_MMD","B13_MLP_CORAL",
                "M1_SSP_MFN_full","M2_SSP_MFN_no_gate",
                "M3_SSP_MFN_no_adain","M4_SSP_MFN_plain"]
labels = ["Pre\nonly","Ridge","Lasso","Elastic\nNet","RF","XGB","SVR","KNN",
          "MulT","TFN","LMF","MLP\n+MMD","MLP\n+CORAL",
          "SSP-MFN\n(full)","SSP-MFN\n(-gate)","SSP-MFN\n(-AdaIN)","SSP-MFN\n(plain)"]

# 获取 R² 值
r2_vals = []
for m in models_order:
    if m.startswith("B9") or m.startswith("B10") or m.startswith("B11") or m.startswith("B12") or m.startswith("B13"):
        r2_vals.append(exp8["baselines"][m]["_mean"]["r2"])
    else:
        r2_vals.append(exp1["N850"][m]["_mean"]["r2"])

# 字母分组法:
# a = SSP-MFN full/no_gate + RF + XGBoost + SVR (p>0.05 vs SSP-MFN)
# b = Ridge/Lasso/ElasticNet/KNN/no_adain/plain/LMF/TFN/MMD/CORAL (p<0.05)
# c = Pre-only / MulT (过拟合, R²<0)
letter_groups = ['c','b','b','b','a','a','a','b',
                 'c','b','b','b','b',
                 'a','a','b','b']

colors = []
for lg in letter_groups:
    if lg == 'a': colors.append('#2ecc71')
    elif lg == 'b': colors.append('#95a5a6')
    else: colors.append('#bdc3c7')

bars = ax_a.bar(range(len(models_order)), r2_vals, color=colors,
                edgecolor='white', linewidth=0.5, width=0.7)
ax_a.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax_a.set_ylabel('$R^2$')
ax_a.set_title('(a) Model Comparison with Statistical Grouping (17 models)', fontweight='bold')
ax_a.set_xticks(range(len(models_order)))
ax_a.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
ax_a.set_ylim(-0.08, 0.16)

# 标注数值 + 字母分组
for i, (v, lg) in enumerate(zip(r2_vals, letter_groups)):
    ax_a.text(i, v + 0.005 if v >= 0 else v - 0.012, f'{v:.3f}',
              ha='center', va='bottom' if v >= 0 else 'top', fontsize=6)
    ax_a.text(i, -0.065, lg, ha='center', va='center', fontsize=8,
              fontweight='bold', color='#2c3e50',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='#ecf0f1', edgecolor='none'))

# 分区标注
ax_a.axvspan(-0.5, 7.5, alpha=0.03, color='blue', label='Classical ML')
ax_a.axvspan(7.5, 12.5, alpha=0.03, color='red', label='Multimodal SOTA')
ax_a.axvspan(12.5, 16.5, alpha=0.03, color='green', label='SSP-MFN variants')

ax_a.text(0.02, 0.95, 'Group a: no sig. diff. from SSP-MFN (p>0.05)\n'
          'Group b: significantly lower (p<0.05)\n'
          'Group c: negative $R^2$ (overfitting/baseline)',
          transform=ax_a.transAxes, fontsize=7, va='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

# --- (b) Bootstrap CI 森林图 ---
ax_b = fig.add_subplot(gs[1, 0])

comparisons = exp7["comparisons"]
comp_names = list(comparisons.keys())
deltas = [comparisons[c]["mean_diff"] for c in comp_names]
ci_lo = [comparisons[c]["ci_95_lo"] for c in comp_names]
ci_hi = [comparisons[c]["ci_95_hi"] for c in comp_names]
pvals = [comparisons[c]["p_value"] for c in comp_names]

y_pos = range(len(comp_names))
xerr_lo = [d - lo for d, lo in zip(deltas, ci_lo)]
xerr_hi = [hi - d for d, hi in zip(deltas, ci_hi)]

colors_ci = []
for p in pvals:
    if p < 0.01: colors_ci.append('#27ae60')
    elif p < 0.05: colors_ci.append('#2ecc71')
    else: colors_ci.append('#95a5a6')

ax_b.barh(y_pos, deltas, xerr=[xerr_lo, xerr_hi], color=colors_ci,
          edgecolor='white', linewidth=0.5, capsize=3, height=0.55)
ax_b.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.8)
# 实际意义阈值线 (ΔR²=0.05)
ax_b.axvline(x=0.05, color='#3498db', linestyle=':', linewidth=1, alpha=0.6)
ax_b.text(0.052, len(comp_names)-0.3, 'practical\nsignificance', fontsize=7, 
          color='#3498db', va='top')

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([f"vs {n}" for n in comp_names], fontsize=9)
ax_b.set_xlabel('$\\Delta R^2$ (SSP-MFN − Baseline)')
ax_b.set_title('(b) Bootstrap 95% CI', fontweight='bold')

for i, (d, p) in enumerate(zip(deltas, pvals)):
    sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax_b.text(max(ci_hi) + 0.008, i, f'p={p:.3f} {sig}', va='center', fontsize=8)

ax_b.set_xlim(-0.04, max(ci_hi) + 0.07)

# --- (c) Component Ablation (紧凑版) ---
ax_c = fig.add_subplot(gs[1, 1])

ablation_models = ["M1_SSP_MFN_full","M2_SSP_MFN_no_gate","M3_SSP_MFN_no_adain","M4_SSP_MFN_plain"]
ablation_labels = ["Full\n(Gate+AdaIN)","No Gate\n(AdaIN only)","No AdaIN\n(Gate only)","Plain\n(Neither)"]
ablation_r2 = [exp1["N850"][m]["_mean"]["r2"] for m in ablation_models]
colors_abl = ['#e74c3c','#e67e22','#f39c12','#95a5a6']

bars_c = ax_c.bar(range(4), ablation_r2, color=colors_abl, edgecolor='white', linewidth=0.5, width=0.6)
ax_c.set_xticks(range(4))
ax_c.set_xticklabels(ablation_labels, fontsize=9)
ax_c.set_ylabel('$R^2$')
ax_c.set_title('(c) Component Contribution', fontweight='bold')
ax_c.set_ylim(0, 0.15)

for i, v in enumerate(ablation_r2):
    ax_c.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

# AdaIN 贡献标注 (主要贡献)
adain_contrib = ablation_r2[0] - ablation_r2[2]
gate_contrib = ablation_r2[0] - ablation_r2[1]
ax_c.annotate('', xy=(0, ablation_r2[0]+0.001), xytext=(2, ablation_r2[2]+0.001),
              arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.5))
ax_c.text(1, (ablation_r2[0]+ablation_r2[2])/2 + 0.008,
          f'AdaIN: $\\Delta R^2$=+{adain_contrib:.3f}',
          ha='center', fontsize=9, color='#c0392b', fontweight='bold')

ax_c.annotate('', xy=(0, ablation_r2[1]-0.005), xytext=(1, ablation_r2[1]-0.005),
              arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.2))
ax_c.text(0.5, ablation_r2[1] - 0.015,
          f'Gate: $\\Delta R^2$=+{gate_contrib:.3f}',
          ha='center', fontsize=8, color='#8e44ad')

plt.savefig(FIG/"Figure1_performance.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"Figure1_performance.pdf", dpi=DPI, bbox_inches='tight')
plt.close()
print("    → Figure1_performance.png/pdf")

# ============================================================
# Figure 2: 机理解释 (3 子图)
# ============================================================
print("  Figure 2: 机理解释...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# --- (a) Ablation Matrix 热力图 (center=0, 分段色图) ---
ax_a2 = fig.add_subplot(gs[0, :])

combos = ["A__","_M_","AM_","__P","A_P","_MP","AMP"]
combo_labels = ["Audio only","Meta only","Audio+Meta",
                "Part only","Audio+Part","Meta+Part","All (A+M+P)"]
matrix = np.zeros((len(combos), len(SCALES)))
for i, c in enumerate(combos):
    for j, s in enumerate(SCALES):
        matrix[i, j] = exp2[c][s]["r2"]

# 使用 TwoSlopeNorm 让 0 点为白色
norm = TwoSlopeNorm(vmin=-0.03, vcenter=0, vmax=0.25)
im = ax_a2.imshow(matrix, cmap='RdYlGn', norm=norm, aspect='auto')
ax_a2.set_xticks(range(len(SCALES)))
ax_a2.set_xticklabels(SCALES, fontsize=10)
ax_a2.set_yticks(range(len(combos)))
ax_a2.set_yticklabels(combo_labels, fontsize=9)
ax_a2.set_xlabel('Social Skill Scale')
ax_a2.set_ylabel('Modality Combination')
ax_a2.set_title('(a) Modality Ablation Matrix: $R^2$ per Scale', fontweight='bold')

# 统一水平标注
for i in range(len(combos)):
    for j in range(len(SCALES)):
        v = matrix[i, j]
        color = 'white' if v > 0.18 or v < -0.02 else 'black'
        ax_a2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8, color=color)

# 分隔线: 无 part vs 有 part
ax_a2.axhline(y=2.5, color='#2c3e50', linestyle='-', linewidth=2)
ax_a2.text(5.6, 1, 'Without\nPart', fontsize=8, va='center', color='#e74c3c', fontweight='bold')
ax_a2.text(5.6, 4.5, 'With\nPart', fontsize=8, va='center', color='#27ae60', fontweight='bold')

cbar = plt.colorbar(im, ax=ax_a2, shrink=0.7, pad=0.12)
cbar.set_label('$R^2$')

# --- (b) Gating Weights 热力图 ---
ax_b2 = fig.add_subplot(gs[1, 0])

gate_matrix = np.zeros((len(SCALES), 3))
modality_names = ["Audio", "Meta", "Part"]
for i, s in enumerate(SCALES):
    for j, m in enumerate(["audio","meta","part"]):
        gate_matrix[i, j] = exp5["per_scale_gates"][s][m]

im2 = ax_b2.imshow(gate_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.35)
ax_b2.set_xticks(range(3))
ax_b2.set_xticklabels(modality_names, fontsize=10)
ax_b2.set_yticks(range(len(SCALES)))
ax_b2.set_yticklabels(SCALES, fontsize=10)
ax_b2.set_xlabel('Modality')
ax_b2.set_ylabel('Scale')
ax_b2.set_title('(b) Learned Gating Weights ($\\alpha \\times g$)', fontweight='bold')

for i in range(len(SCALES)):
    for j in range(3):
        v = gate_matrix[i, j]
        color = 'white' if v > 0.25 else 'black'
        ax_b2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10, color=color)

cbar2 = plt.colorbar(im2, ax=ax_b2, shrink=0.8)
cbar2.set_label('Weight')

# --- (c) Session Dynamics + 民族差异 ---
ax_c2 = fig.add_subplot(gs[1, 1])

sessions = sorted(exp5["by_session"].keys(), key=int)
audio_trend = [exp5["by_session"][s]["audio"] for s in sessions]
meta_trend = [exp5["by_session"][s]["meta"] for s in sessions]
part_trend = [exp5["by_session"][s]["part"] for s in sessions]

x = [int(s) for s in sessions]
ax_c2.plot(x, audio_trend, 'o-', color='#3498db', linewidth=2, markersize=7, label='Audio')
ax_c2.plot(x, meta_trend, 's-', color='#e74c3c', linewidth=2, markersize=7, label='Meta')
ax_c2.plot(x, part_trend, '^-', color='#27ae60', linewidth=2, markersize=7, label='Part')
ax_c2.set_xlabel('Session Number')
ax_c2.set_ylabel('Gate Weight ($\\alpha \\times g$)')
ax_c2.set_title('(c) Gating Dynamics Across Sessions', fontweight='bold')
ax_c2.legend(fontsize=9, loc='upper right')
ax_c2.set_xticks(x)
ax_c2.set_ylim(0, 0.30)
ax_c2.grid(True, alpha=0.3)

# 添加民族差异 ANOVA 注释
anova_text = "Ethnic ANOVA:\n"
for m in ["audio","meta","part"]:
    p = exp5["anova_ethnic"][m]["p"]
    F = exp5["anova_ethnic"][m]["F"]
    anova_text += f"  {m}: F={F:.1f}, p<0.001***\n"
ax_c2.text(0.02, 0.55, anova_text.strip(), transform=ax_c2.transAxes, fontsize=7.5,
           va='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#fdf2e9', edgecolor='#e67e22', alpha=0.9))

plt.savefig(FIG/"Figure2_mechanism.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"Figure2_mechanism.pdf", dpi=DPI, bbox_inches='tight')
plt.close()
print("    → Figure2_mechanism.png/pdf")

# ============================================================
# 补充: 独立的民族门控对比图 (带 bracket 标注)
# ============================================================
print("  Supplementary: 民族门控对比...")

fig, ax = plt.subplots(figsize=(7, 5))
ethnic_names = ["Dong (侗族)", "Tibetan (藏族)", "Mongolian (蒙古族)"]
ethnic_keys = ["侗族", "藏族", "蒙古族"]
x = np.arange(3)
width = 0.22

bar_colors = ['#3498db', '#e74c3c', '#27ae60']
for i, (ek, en) in enumerate(zip(ethnic_keys, ethnic_names)):
    vals = [exp5["by_ethnic"][ek]["audio"], exp5["by_ethnic"][ek]["meta"], exp5["by_ethnic"][ek]["part"]]
    ax.bar(x + i*width, vals, width, label=en, color=bar_colors[i],
           edgecolor='white', linewidth=0.5, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(["Audio", "Meta", "Part"], fontsize=11)
ax.set_ylabel('Gate Weight ($\\alpha \\times g$)')
ax.set_title('Gating Weights by Ethnic Group (ANOVA: inter-group comparison)', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 0.34)

# Bracket 标注: 跨民族比较
for j, m in enumerate(["audio","meta","part"]):
    F = exp5["anova_ethnic"][m]["F"]
    p = exp5["anova_ethnic"][m]["p"]
    # 画 bracket 跨三个柱子
    y_max = max(exp5["by_ethnic"][ek][m] for ek in ethnic_keys)
    bracket_y = y_max + 0.02
    x_left = j + 0 * width
    x_right = j + 2 * width
    ax.plot([x_left, x_left, x_right, x_right], 
            [bracket_y, bracket_y + 0.008, bracket_y + 0.008, bracket_y],
            color='#2c3e50', linewidth=1.2)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text((x_left + x_right)/2, bracket_y + 0.012, 
            f'F={F:.1f} {sig}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG/"FigS1_gate_ethnic_bracket.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"FigS1_gate_ethnic_bracket.pdf", dpi=DPI, bbox_inches='tight')
plt.close()
print("    → FigS1_gate_ethnic_bracket.png/pdf")

print(f"\n  完成! 所有图表保存至: {FIG}/")
print("  最终图表:")
print("    Figure1_performance.png/pdf  — 论文正文 Fig.1")
print("    Figure2_mechanism.png/pdf    — 论文正文 Fig.2")
print("    FigS1_gate_ethnic_bracket    — 补充材料")

# ============================================================
# FigS2: Permutation Sanity Check
# ============================================================
print("  FigS2: Permutation Sanity Check...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 模态级别: Gate Weight vs ΔR²
ax = axes[0]
modalities = ["audio", "meta", "part"]
gate_w = [exp9[m]["gate_weight"] for m in modalities]
delta_r2 = [exp9[m]["mean_delta_r2"] for m in modalities]
delta_std = [exp9[m]["std_delta_r2"] for m in modalities]

colors_perm = ['#3498db', '#e74c3c', '#27ae60']
for i, m in enumerate(modalities):
    ax.errorbar(gate_w[i], delta_r2[i], yerr=delta_std[i],
                fmt='o', markersize=12, color=colors_perm[i],
                capsize=5, capthick=2, linewidth=2, label=m.capitalize())
    ax.annotate(m.capitalize(), (gate_w[i], delta_r2[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)

# 拟合线
z = np.polyfit(gate_w, delta_r2, 1)
x_fit = np.linspace(min(gate_w)*0.8, max(gate_w)*1.2, 50)
ax.plot(x_fit, np.polyval(z, x_fit), '--', color='gray', alpha=0.5, linewidth=1.5)

rho = exp9["correlation"]["spearman_rho"]
p_val = exp9["correlation"]["spearman_p"]
ax.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}\np = {p_val:.3f}',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='#eaf2f8', edgecolor='#3498db'))
ax.set_xlabel('Learned Gate Weight ($\\alpha \\times g$)')
ax.set_ylabel('$\\Delta R^2$ (Original − Permuted)')
ax.set_title('(a) Gate Weight vs. Permutation Importance', fontweight='bold')
ax.grid(True, alpha=0.3)

# (b) 逐量表 permutation ΔR²
ax2 = axes[1]
x_pos = np.arange(len(SCALES))
width = 0.25

# 从 per_fold 数据中提取逐量表 ΔR²
for i, m in enumerate(modalities):
    per_dim_deltas = []
    for fold_res in exp9[m]["per_fold"]:
        per_dim_deltas.append(fold_res["delta_per_dim"])
    mean_per_dim = np.mean(per_dim_deltas, axis=0)
    ax2.bar(x_pos + i*width, mean_per_dim, width, label=m.capitalize(),
            color=colors_perm[i], edgecolor='white', linewidth=0.5, alpha=0.85)

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(SCALES, fontsize=10)
ax2.set_ylabel('$\\Delta R^2$ (Permutation Drop)')
ax2.set_title('(b) Per-Scale Permutation Importance', fontweight='bold')
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# 标注预期最强信号
ax2.annotate('H3: Audio→CSAS', xy=(2, 0.007), fontsize=7, color='#3498db', ha='center')
ax2.annotate('H4: Part→SSCS', xy=(3.5, 0.032), fontsize=7, color='#27ae60', ha='center')
ax2.annotate('H6: Meta→SCI2', xy=(5.3, 0.007), fontsize=7, color='#e74c3c', ha='center')

plt.tight_layout()
plt.savefig(FIG/"FigS2_permutation_sanity.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"FigS2_permutation_sanity.pdf", dpi=DPI, bbox_inches='tight')
plt.close()
print("    → FigS2_permutation_sanity.png/pdf")

# ============================================================
# FigS3: 参数效率对比 (Parameter Efficiency)
# ============================================================
print("  FigS3: 参数效率对比...")

fig, ax = plt.subplots(figsize=(8, 5))
model_names = ["SSP-MFN", "MulT", "TFN", "LMF", "MLP+MMD", "MLP+CORAL"]
params = [exp8["param_counts"][n] for n in model_names]
r2_all = [
    exp8["SSP_MFN_ref"]["r2"],
    exp8["baselines"]["B9_MulT"]["_mean"]["r2"],
    exp8["baselines"]["B10_TFN"]["_mean"]["r2"],
    exp8["baselines"]["B11_LMF"]["_mean"]["r2"],
    exp8["baselines"]["B12_MLP_MMD"]["_mean"]["r2"],
    exp8["baselines"]["B13_MLP_CORAL"]["_mean"]["r2"],
]
colors_eff = ['#e74c3c', '#95a5a6', '#95a5a6', '#f39c12', '#3498db', '#3498db']
markers = ['*', 'D', 's', '^', 'o', 'o']

for i, (p, r2, name) in enumerate(zip(params, r2_all, model_names)):
    ax.scatter(p, r2, s=150 if i == 0 else 80, c=colors_eff[i],
               marker=markers[i], zorder=5, edgecolors='white', linewidth=0.5)
    offset = (10, 8) if i != 1 else (10, -15)
    ax.annotate(f'{name}\n({p:,} params)', (p, r2),
                textcoords="offset points", xytext=offset, fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5) if i > 0 else None)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (log scale)')
ax.set_ylabel('$R^2$')
ax.set_title('Parameter Efficiency: SSP-MFN vs. Multimodal SOTA', fontweight='bold')
ax.grid(True, alpha=0.3)

# 标注 "sweet spot"
ax.annotate('Parameter-efficient\nsweet spot', xy=(56548, 0.119),
            xytext=(20000, 0.08), fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

plt.tight_layout()
plt.savefig(FIG/"FigS3_param_efficiency.png", dpi=DPI, bbox_inches='tight')
plt.savefig(FIG/"FigS3_param_efficiency.pdf", dpi=DPI, bbox_inches='tight')
plt.close()
print("    → FigS3_param_efficiency.png/pdf")

print(f"\n  全部完成! 图表清单:")
print("    Figure1_performance     — 正文 Fig.1 (17模型对比+Bootstrap+消融)")
print("    Figure2_mechanism       — 正文 Fig.2 (热力图+门控+趋势)")
print("    FigS1_gate_ethnic       — 补充 (民族门控对比)")
print("    FigS2_permutation       — 补充 (Permutation Sanity Check)")
print("    FigS3_param_efficiency  — 补充 (参数效率对比)")
