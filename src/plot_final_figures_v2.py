import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测")
RES = ROOT / "实验/results"
FIG = ROOT / "实验/figures_v2"; FIG.mkdir(parents=True, exist_ok=True)
DPI = 600

with open(RES/"EXP1_S6_full.json") as f: exp1 = json.load(f)
with open(RES/"EXP2_ablation_matrix.json") as f: exp2 = json.load(f)
with open(RES/"EXP5_gate_weights.json") as f: exp5 = json.load(f)
with open(RES/"EXP7_bootstrap_ci.json") as f: exp7 = json.load(f)
with open(RES/"EXP8_supplementary_baselines.json") as f: exp8 = json.load(f)
with open(RES/"EXP9_permutation_sanity.json") as f: exp9 = json.load(f)

SCALES = ["ICS","IRI","CSAS","SSCS","IOS","SCI2"]

C_OURS = '#2E86AB'
C_OURS_LIGHT = '#7EC8E3'
C_SOTA = '#E8871E'
C_SOTA_LIGHT = '#F5B97A'
C_ML = '#7B7B7B'
C_ML_LIGHT = '#B5B5B5'
C_NEG = '#D4D4D4'
C_AUDIO = '#3498db'
C_META = '#e74c3c'
C_PART = '#27ae60'


# ============================================================
# Figure 1: Performance Evidence (3 subplots)
# ============================================================
print("  Figure 1: Performance Evidence ...")

classical_ml = [
    ("SVR",       "B7_SVR",       exp1["N850"]["B7_SVR"]["_mean"]["r2"]),
    ("RF",        "B5_RF",        exp1["N850"]["B5_RF"]["_mean"]["r2"]),
    ("XGBoost",   "B6_XGBoost",   exp1["N850"]["B6_XGBoost"]["_mean"]["r2"]),
    ("Lasso",     "B3_Lasso",     exp1["N850"]["B3_Lasso"]["_mean"]["r2"]),
    ("ElasticNet","B4_ElasticNet", exp1["N850"]["B4_ElasticNet"]["_mean"]["r2"]),
    ("Pre-only",  "B1_pre_only",  exp1["N850"]["B1_pre_only"]["_mean"]["r2"]),
    ("KNN",       "B8_KNN",       exp1["N850"]["B8_KNN"]["_mean"]["r2"]),
    ("Ridge",     "B2_Ridge",     exp1["N850"]["B2_Ridge"]["_mean"]["r2"]),
]

multimodal_sota = [
    ("LMF",       "B11_LMF",      exp8["baselines"]["B11_LMF"]["_mean"]["r2"]),
    ("TFN",       "B10_TFN",      exp8["baselines"]["B10_TFN"]["_mean"]["r2"]),
    ("MLP+CORAL", "B13_MLP_CORAL", exp8["baselines"]["B13_MLP_CORAL"]["_mean"]["r2"]),
    ("MLP+MMD",   "B12_MLP_MMD",  exp8["baselines"]["B12_MLP_MMD"]["_mean"]["r2"]),
    ("MulT",      "B9_MulT",      exp8["baselines"]["B9_MulT"]["_mean"]["r2"]),
]

ours_variants = [
    ("SSP-MFN (full)",    "M1", exp1["N850"]["M1_SSP_MFN_full"]["_mean"]["r2"]),
    ("SSP-MFN (-gate)",   "M2", exp1["N850"]["M2_SSP_MFN_no_gate"]["_mean"]["r2"]),
    ("SSP-MFN (-AdaIN)",  "M3", exp1["N850"]["M3_SSP_MFN_no_adain"]["_mean"]["r2"]),
    ("SSP-MFN (plain)",   "M4", exp1["N850"]["M4_SSP_MFN_plain"]["_mean"]["r2"]),
]

classical_ml.sort(key=lambda x: x[2], reverse=True)
multimodal_sota.sort(key=lambda x: x[2], reverse=True)
ours_variants.sort(key=lambda x: x[2], reverse=True)

letter_map = {
    "B7_SVR": "a", "B5_RF": "a", "B6_XGBoost": "a",
    "B3_Lasso": "b", "B4_ElasticNet": "b", "B1_pre_only": "c",
    "B8_KNN": "b", "B2_Ridge": "b",
    "B11_LMF": "b", "B10_TFN": "b", "B13_MLP_CORAL": "b",
    "B12_MLP_MMD": "b", "B9_MulT": "c",
    "M1": "a", "M2": "a", "M3": "b", "M4": "b",
}

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.45,
                      height_ratios=[1.3, 1])

# --- (a) Horizontal Bar Chart ---
ax_a = fig.add_subplot(gs[0, :])

all_models = []
all_r2 = []
all_colors = []
all_edge = []
group_boundaries = []

for name, key, r2 in ours_variants:
    all_models.append(name)
    all_r2.append(r2)
    lg = letter_map[key]
    if lg == 'a':
        all_colors.append(C_OURS)
    else:
        all_colors.append(C_OURS_LIGHT)
    all_edge.append(C_OURS)

group_boundaries.append(len(all_models))

for name, key, r2 in multimodal_sota:
    all_models.append(name)
    all_r2.append(r2)
    lg = letter_map[key]
    if lg == 'c':
        all_colors.append(C_NEG)
    else:
        all_colors.append(C_SOTA_LIGHT)
    all_edge.append(C_SOTA)

group_boundaries.append(len(all_models))

for name, key, r2 in classical_ml:
    all_models.append(name)
    all_r2.append(r2)
    lg = letter_map[key]
    if lg == 'a':
        all_colors.append(C_ML)
    elif lg == 'c':
        all_colors.append(C_NEG)
    else:
        all_colors.append(C_ML_LIGHT)
    all_edge.append(C_ML)

y_pos = np.arange(len(all_models))

bars = ax_a.barh(y_pos, all_r2, color=all_colors,
                 edgecolor=[e for e in all_edge], linewidth=0.6,
                 height=0.65, alpha=0.9)

for i, bar in enumerate(bars):
    if i < len(ours_variants):
        bar.set_linewidth(1.8)
        bar.set_edgecolor('#1a5276')

ax_a.axvline(x=0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.6)

for b in group_boundaries:
    ax_a.axhline(y=b - 0.5, color='#999999', linestyle='--',
                 linewidth=0.5, alpha=0.5)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(all_models, fontsize=8.5)
ax_a.set_xlabel('Mean $R^2$ (6-fold CV, N=850)')
ax_a.set_title('(a) Model Comparison with Statistical Grouping',
               fontweight='bold', fontsize=11)
ax_a.set_xlim(-0.06, 0.16)
ax_a.invert_yaxis()

for i, (v, model_name) in enumerate(zip(all_r2, all_models)):
    key = None
    for nm, k, r2 in ours_variants + multimodal_sota + classical_ml:
        if nm == model_name:
            key = k
            break
    lg = letter_map.get(key, '')
    offset = 0.004 if v >= 0 else -0.004
    ha = 'left' if v >= 0 else 'right'
    ax_a.text(v + offset, i, f'{v:.3f} ({lg})',
              va='center', ha=ha, fontsize=7.5,
              fontweight='bold' if lg == 'a' else 'normal')

ax_a.text(0.98, 0.02,
          'Group a: p > 0.05 vs SSP-MFN\n'
          'Group b: p < 0.05\n'
          'Group c: $R^2$ < 0',
          transform=ax_a.transAxes, fontsize=7.5, va='bottom', ha='right',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    alpha=0.9, edgecolor='#cccccc'))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C_OURS, edgecolor=C_OURS, label='SSP-MFN variants'),
    Patch(facecolor=C_SOTA_LIGHT, edgecolor=C_SOTA, label='Multimodal SOTA'),
    Patch(facecolor=C_ML, edgecolor=C_ML, label='Classical ML'),
]
ax_a.legend(handles=legend_elements, loc='lower right', fontsize=8,
            framealpha=0.9, edgecolor='#cccccc',
            bbox_to_anchor=(0.98, 0.18))

ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# --- (b) Bootstrap CI Forest Plot ---
ax_b = fig.add_subplot(gs[1, 0])

comparisons = exp7["comparisons"]
comp_order = ["SVR","RF","XGBoost","Lasso","ElasticNet","KNN","Ridge"]
deltas = [comparisons[c]["mean_diff"] for c in comp_order]
ci_lo = [comparisons[c]["ci_95_lo"] for c in comp_order]
ci_hi = [comparisons[c]["ci_95_hi"] for c in comp_order]
pvals = [comparisons[c]["p_value"] for c in comp_order]

y_pos_b = np.arange(len(comp_order))

for i, (d, lo, hi, p) in enumerate(zip(deltas, ci_lo, ci_hi, pvals)):
    color = C_OURS if p < 0.05 else C_ML_LIGHT
    ax_b.plot([lo, hi], [i, i], color=color, linewidth=2.5, solid_capstyle='round')
    ax_b.plot(d, i, 'o', color=color, markersize=7, markeredgecolor='white',
              markeredgewidth=0.8, zorder=5)

ax_b.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)

ax_b.set_yticks(y_pos_b)
ax_b.set_yticklabels([f'vs {n}' for n in comp_order], fontsize=8.5)
ax_b.set_xlabel('$\\Delta R^2$ (SSP-MFN $-$ Baseline)')
ax_b.set_title('(b) Bootstrap 95% CI (1000 resamples)', fontweight='bold')

for i, (d, p) in enumerate(zip(deltas, pvals)):
    sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax_b.text(max(ci_hi) + 0.012, i, f'p={p:.3f} {sig}',
              va='center', fontsize=7,
              fontweight='bold' if p < 0.05 else 'normal',
              color='#2c3e50' if p < 0.05 else '#999999')

ax_b.set_xlim(-0.04, max(ci_hi) + 0.10)
ax_b.invert_yaxis()
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# --- (c) Component Ablation ---
ax_c = fig.add_subplot(gs[1, 1])

ablation_models = ["M1_SSP_MFN_full","M2_SSP_MFN_no_gate",
                   "M3_SSP_MFN_no_adain","M4_SSP_MFN_plain"]
ablation_labels = ["Full\n(Gate+AdaIN)","$-$Gate\n(AdaIN only)",
                   "$-$AdaIN\n(Gate only)","Plain\n(Neither)"]
ablation_r2 = [exp1["N850"][m]["_mean"]["r2"] for m in ablation_models]

colors_abl = [C_OURS, '#5BA3C9', C_OURS_LIGHT, C_ML_LIGHT]

bars_c = ax_c.bar(range(4), ablation_r2, color=colors_abl,
                  edgecolor='white', linewidth=0.8, width=0.6)
ax_c.set_xticks(range(4))
ax_c.set_xticklabels(ablation_labels, fontsize=8.5)
ax_c.set_ylabel('Mean $R^2$')
ax_c.set_title('(c) Component Contribution', fontweight='bold')

for i, v in enumerate(ablation_r2):
    ax_c.text(i, v + 0.004, f'{v:.4f}', ha='center', fontsize=8.5,
              fontweight='bold' if i == 0 else 'normal')

adain_contrib = ablation_r2[0] - ablation_r2[2]
gate_contrib = ablation_r2[0] - ablation_r2[1]

# AdaIN annotation: place above the bars in clear space
bracket_y_adain = max(ablation_r2) + 0.015
ax_c.plot([0, 0, 2, 2],
          [bracket_y_adain - 0.003, bracket_y_adain, bracket_y_adain, bracket_y_adain - 0.003],
          color='#c0392b', linewidth=1.5)
ax_c.text(1, bracket_y_adain + 0.005,
          f'AdaIN: $\\Delta R^2$=+{adain_contrib:.3f}',
          ha='center', fontsize=8, color='#c0392b', fontweight='bold')

# Gate annotation: side annotation between Full and -Gate bars
mid_y_gate = (ablation_r2[0] + ablation_r2[1]) / 2
ax_c.annotate('', xy=(-0.35, ablation_r2[0] - 0.002), xytext=(-0.35, ablation_r2[1] + 0.002),
              arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.2,
                              shrinkA=0, shrinkB=0))
ax_c.text(-0.55, mid_y_gate,
          f'Gate:\n$\\Delta R^2$\n=+{gate_contrib:.3f}',
          ha='center', va='center', fontsize=6.5, color='#8e44ad')

ax_c.set_xlim(-0.8, 3.5)
ax_c.set_ylim(0, 0.17)

ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

plt.savefig(FIG/"Figure1_performance_v2.png", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.savefig(FIG/"Figure1_performance_v2.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    -> Figure1_performance_v2.png/pdf")

# ============================================================
# Figure 2: Mechanism Explanation (3 subplots)
# ============================================================
print("  Figure 2: Mechanism Explanation ...")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.12,
                      height_ratios=[1.85, 1],
                      width_ratios=[1.00, 0.68, 0.32])

# --- (a) Ablation Matrix Heatmap ---
ax_a2 = fig.add_subplot(gs[0, :])

combos = ["A__","_M_","AM_","__P","A_P","_MP","AMP"]
combo_labels = ["Audio only","Meta only","Audio+Meta",
                "Part only","Audio+Part","Meta+Part","All (A+M+P)"]
matrix = np.zeros((len(combos), len(SCALES)))
for i, c in enumerate(combos):
    for j, s in enumerate(SCALES):
        matrix[i, j] = exp2[c][s]["r2"]

norm = TwoSlopeNorm(vmin=-0.03, vcenter=0, vmax=0.25)
im = ax_a2.imshow(matrix, cmap='RdYlGn', norm=norm, aspect='auto')
ax_a2.set_xticks(range(len(SCALES)))
ax_a2.set_xticklabels(SCALES, fontsize=10)
ax_a2.set_yticks(range(len(combos)))
ax_a2.set_yticklabels(combo_labels, fontsize=9)
ax_a2.set_xlabel('Social Skill Scale')
ax_a2.set_ylabel('Modality Combination')
ax_a2.set_title('(a) Modality Ablation Matrix: $R^2$ per Scale', fontweight='bold')

for i in range(len(combos)):
    for j in range(len(SCALES)):
        v = matrix[i, j]
        color = 'white' if v > 0.18 or v < -0.02 else 'black'
        ax_a2.text(j, i, f'{v:.3f}', ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold' if i == 6 else 'normal')

ax_a2.axhline(y=2.5, color='#2c3e50', linestyle='-', linewidth=2)
ax_a2.text(5.7, 1, 'Without\nPart', fontsize=8, va='center',
           color=C_META, fontweight='bold')
ax_a2.text(5.7, 4.5, 'With\nPart', fontsize=8, va='center',
           color=C_PART, fontweight='bold')

cbar = plt.colorbar(im, ax=ax_a2, shrink=0.7, pad=0.10)
cbar.set_label('$R^2$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# --- (b) Gating Weights Heatmap ---
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
ax_b2.set_title('(b) Learned Gating Weights ($\\alpha \\cdot g$)', fontweight='bold')

for i in range(len(SCALES)):
    for j in range(3):
        v = gate_matrix[i, j]
        color = 'white' if v > 0.25 else 'black'
        ax_b2.text(j, i, f'{v:.3f}', ha='center', va='center',
                   fontsize=10, color=color)

cbar2 = plt.colorbar(im2, ax=ax_b2, shrink=0.8)
cbar2.set_label('Weight', fontsize=9)
cbar2.ax.tick_params(labelsize=8)

# --- (c) Session Dynamics ---
ax_c2 = fig.add_subplot(gs[1, 1])

sessions = sorted(exp5["by_session"].keys(), key=int)
audio_trend = [exp5["by_session"][s]["audio"] for s in sessions]
meta_trend = [exp5["by_session"][s]["meta"] for s in sessions]
part_trend = [exp5["by_session"][s]["part"] for s in sessions]

x = [int(s) for s in sessions]
ax_c2.plot(x, audio_trend, marker='o', linestyle='-', color=C_AUDIO,
           linewidth=2, markersize=7, label='Audio',
           markeredgecolor='white', markeredgewidth=0.8)
ax_c2.plot(x, meta_trend, marker='s', linestyle='--', color=C_META,
           linewidth=2, markersize=7, label='Meta',
           markeredgecolor='white', markeredgewidth=0.8)
ax_c2.plot(x, part_trend, marker='^', linestyle='-.', color=C_PART,
           linewidth=2, markersize=7, label='Part',
           markeredgecolor='white', markeredgewidth=0.8)

ax_c2.set_xlabel('Session Number')
ax_c2.set_ylabel('Gate Weight ($\\alpha \\cdot g$)')
ax_c2.set_title('(c) Gating Dynamics Across Sessions', fontweight='bold')
ax_c2.legend(fontsize=8.5, loc='upper right', framealpha=0.9, edgecolor='#cccccc')
ax_c2.set_xticks(x)
ax_c2.set_ylim(0, 0.32)
ax_c2.grid(True, alpha=0.2, linestyle='-')

anova_text = "Ethnic ANOVA:\n"
for m in ["audio","meta","part"]:
    p = exp5["anova_ethnic"][m]["p"]
    F = exp5["anova_ethnic"][m]["F"]
    anova_text += f"  {m}: F={F:.1f}, p<0.001***\n"
ax_c2.text(0.02, 0.38, anova_text.strip(), transform=ax_c2.transAxes, fontsize=7,
           va='top', family='monospace',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdf2e9',
                     edgecolor='#e67e22', alpha=0.9))

ax_c2.spines['top'].set_visible(False)
ax_c2.spines['right'].set_visible(False)

plt.savefig(FIG/"Figure2_mechanism_v2.png", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.savefig(FIG/"Figure2_mechanism_v2.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    -> Figure2_mechanism_v2.png/pdf")

# ============================================================
# FigS1: Ethnic Gating Comparison (optimised)
# ============================================================
print("  FigS1: Ethnic gating comparison ...")

fig, ax = plt.subplots(figsize=(7, 4.5))
ethnic_names = ["Dong", "Tibetan", "Mongolian"]
ethnic_keys = ["侗族", "藏族", "蒙古族"]
modality_labels = ["Audio", "Meta", "Part"]
modality_keys = ["audio", "meta", "part"]

x = np.arange(3)  # 3 modalities
width = 0.24
offsets = [-width, 0, width]

# Use distinct but harmonious colors for ethnic groups
ethnic_colors = ['#4C72B0', '#DD8452', '#55A868']
ethnic_hatches = ['///', '\\\\\\', 'xxx']

for i, (ek, en, col, hatch) in enumerate(zip(ethnic_keys, ethnic_names, ethnic_colors, ethnic_hatches)):
    vals = [exp5["by_ethnic"][ek][m] for m in modality_keys]
    ax.bar(x + offsets[i], vals, width, label=en, color=col,
           edgecolor='white', linewidth=0.8, alpha=0.85, hatch=hatch)

ax.set_xticks(x)
ax.set_xticklabels(modality_labels, fontsize=11)
ax.set_ylabel('Gate Weight ($\\alpha \\cdot g$)', fontsize=10)
ax.set_title('Gating Weights by Ethnic Group', fontweight='bold', fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc', loc='upper left')
ax.set_ylim(0, 0.36)

# Add ANOVA brackets with proper spacing
for j, m in enumerate(modality_keys):
    F = exp5["anova_ethnic"][m]["F"]
    p = exp5["anova_ethnic"][m]["p"]
    y_max = max(exp5["by_ethnic"][ek][m] for ek in ethnic_keys)
    bracket_y = y_max + 0.015
    x_left = j + offsets[0]
    x_right = j + offsets[2]
    ax.plot([x_left, x_left, x_right, x_right],
            [bracket_y, bracket_y + 0.008, bracket_y + 0.008, bracket_y],
            color='#2c3e50', linewidth=1.0)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text((x_left + x_right)/2, bracket_y + 0.012,
            f'F={F:.1f}{sig}', ha='center', fontsize=8)

# Add value labels on top of each bar
for i, (ek, en) in enumerate(zip(ethnic_keys, ethnic_names)):
    for j, m in enumerate(modality_keys):
        val = exp5["by_ethnic"][ek][m]
        ax.text(j + offsets[i], val + 0.004, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6.5, color='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.15, linestyle='-')

plt.tight_layout()
plt.savefig(FIG/"FigS1_gate_ethnic_v2.png", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.savefig(FIG/"FigS1_gate_ethnic_v2.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    -> FigS1_gate_ethnic_v2.png/pdf")

# ============================================================
# FigS2: Permutation Sanity Check
# ============================================================
print("  FigS2: Permutation Sanity Check ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'wspace': 0.40})

modalities = ["audio", "meta", "part"]
mod_labels = {"audio": "Audio (MERT)", "meta": "Meta", "part": "Participant"}
colors_perm = {'audio': '#4C72B0', 'meta': '#DD8452', 'part': '#55A868'}
hatches_perm = {'audio': '///', 'meta': '\\\\\\', 'part': 'xxx'}

gate_w = [exp9[m]["gate_weight"] for m in modalities]
delta_r2 = [exp9[m]["mean_delta_r2"] for m in modalities]
delta_std = [exp9[m]["std_delta_r2"] for m in modalities]

# --- (a) Dual-metric horizontal bar chart ---
ax = axes[0]
y_pos_perm = np.arange(len(modalities))
bar_h = 0.35

bars_gate = ax.barh(y_pos_perm + bar_h/2, gate_w, bar_h,
                    color=[colors_perm[m] for m in modalities],
                    edgecolor='white', linewidth=0.8, alpha=0.9,
                    label='Gate Weight ($\\alpha \\cdot g$)')

ax2_top = ax.twiny()
bars_delta = ax2_top.barh(y_pos_perm - bar_h/2, delta_r2, bar_h,
                          xerr=delta_std, capsize=4,
                          color=[colors_perm[m] for m in modalities],
                          edgecolor=[colors_perm[m] for m in modalities],
                          linewidth=1.2, alpha=0.35,
                          hatch='///',
                          label='$\\Delta R^2$ (Permutation)')

for i, m in enumerate(modalities):
    ax.text(gate_w[i] + 0.005, i + bar_h/2, f'{gate_w[i]:.3f}',
            va='center', ha='left', fontsize=9, fontweight='bold',
            color=colors_perm[m])
    ax2_top.text(delta_r2[i] + delta_std[i] + 0.0008, i - bar_h/2,
                 f'{delta_r2[i]:.4f}',
                 va='center', ha='left', fontsize=8.5,
                 color=colors_perm[m])

ax.set_yticks(y_pos_perm)
ax.set_yticklabels([mod_labels[m] for m in modalities], fontsize=10, fontweight='bold')
ax.set_xlabel('Gate Weight ($\\alpha \\cdot g$)', fontsize=10, color='#2c3e50')
ax2_top.set_xlabel('$\\Delta R^2$ (Permutation Drop)', fontsize=10, color='#7f8c8d')
ax.set_title('(a) Gate–Importance Consistency', fontweight='bold', fontsize=11)
ax.set_xlim(0, 0.33)
ax2_top.set_xlim(0, 0.016)

rho = exp9["correlation"]["spearman_rho"]
ax.text(0.02, 0.05,
        f'Spearman $\\rho$ = {rho:.3f}\n(n=3; rank-order\nconsistency check)',
        transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa',
                  edgecolor='#adb5bd', alpha=0.95, linewidth=0.8))

ax.text(0.97, 0.50,
        'Solid bars = Gate Weight (bottom axis)\nHatched bars = $\\Delta R^2$ (top axis)',
        transform=ax.transAxes, fontsize=7, va='center', ha='right',
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95, linewidth=0.6))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2_top.spines['right'].set_visible(False)
ax.invert_yaxis()
ax2_top.invert_yaxis()

# --- (b) Per-Scale Permutation Importance with gate weight overlay ---
ax2 = axes[1]
x_pos = np.arange(len(SCALES))
width = 0.25

per_dim_means = {}
for m in modalities:
    per_dim_deltas = []
    for fold_res in exp9[m]["per_fold"]:
        per_dim_deltas.append(fold_res["delta_per_dim"])
    per_dim_means[m] = np.mean(per_dim_deltas, axis=0)

for i, m in enumerate(modalities):
    mean_per_dim = per_dim_means[m]
    bars = ax2.bar(x_pos + i*width, mean_per_dim, width,
                   label=mod_labels[m], color=colors_perm[m],
                   edgecolor='white', linewidth=0.5, alpha=0.85,
                   hatch=hatches_perm[m])
    for j, (bar, val) in enumerate(zip(bars, mean_per_dim)):
        if val > 0.006:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=6, color=colors_perm[m], fontweight='bold')

ax2_right = ax2.twinx()
gate_per_scale = {}
for s_idx, s in enumerate(SCALES):
    for m in modalities:
        if m not in gate_per_scale:
            gate_per_scale[m] = []
        gate_per_scale[m].append(exp5["per_scale_gates"][s][m])

markers_line = {'audio': 'o', 'meta': 's', 'part': '^'}
for m in modalities:
    ax2_right.plot(x_pos + width, gate_per_scale[m],
                   marker=markers_line[m], linestyle='--', linewidth=1.2,
                   color=colors_perm[m], markersize=5, alpha=0.7,
                   markeredgecolor='white', markeredgewidth=0.5)

ax2_right.set_ylabel('Gate Weight (dashed lines)', fontsize=9, color='#666666')
ax2_right.set_ylim(0, 0.45)
ax2_right.spines['top'].set_visible(False)

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(SCALES, fontsize=10)
ax2.set_ylabel('$\\Delta R^2$ (Permutation Drop)', fontsize=10)
ax2.set_title('(b) Per-Scale Permutation Importance', fontweight='bold', fontsize=11)
ax2.legend(fontsize=8.5, framealpha=0.95, edgecolor='#cccccc', loc='upper left')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.savefig(FIG/"FigS2_permutation_v2.png", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.savefig(FIG/"FigS2_permutation_v2.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    -> FigS2_permutation_v2.png/pdf")

# ============================================================
# FigS3: Parameter Efficiency + Pareto Frontier
# ============================================================
print("  FigS3: Parameter Efficiency + Pareto Frontier ...")

fig, ax = plt.subplots(figsize=(8, 5.5))

model_names_eff = ["SSP-MFN", "MulT", "TFN", "LMF", "MLP+MMD", "MLP+CORAL"]
params = [exp8["param_counts"][n] for n in model_names_eff]
r2_all = [
    exp8["SSP_MFN_ref"]["r2"],
    exp8["baselines"]["B9_MulT"]["_mean"]["r2"],
    exp8["baselines"]["B10_TFN"]["_mean"]["r2"],
    exp8["baselines"]["B11_LMF"]["_mean"]["r2"],
    exp8["baselines"]["B12_MLP_MMD"]["_mean"]["r2"],
    exp8["baselines"]["B13_MLP_CORAL"]["_mean"]["r2"],
]

colors_eff = [C_OURS, C_ML, C_ML, C_SOTA, C_ML_LIGHT, C_ML_LIGHT]
markers = ['*', 'D', 's', '^', 'o', 'o']
sizes = [350, 90, 90, 110, 80, 80]

for i, (p, r2, name) in enumerate(zip(params, r2_all, model_names_eff)):
    ax.scatter(p, r2, s=sizes[i], c=colors_eff[i],
               marker=markers[i], zorder=5,
               edgecolors='white' if i == 0 else 'none',
               linewidth=1.2 if i == 0 else 0.8)

pareto_pts = sorted(
    [(p, r2) for p, r2 in zip(params, r2_all)],
    key=lambda x: x[0]
)
frontier_p = []
frontier_r = []
best_r2 = -999
for p, r in sorted(pareto_pts, key=lambda x: x[0]):
    if r > best_r2:
        frontier_p.append(p)
        frontier_r.append(r)
        best_r2 = r

if len(frontier_p) > 1:
    ax.plot(frontier_p, frontier_r, '--', color=C_OURS, alpha=0.4,
            linewidth=1.5, zorder=2)

for i, (p, r2, name) in enumerate(zip(params, r2_all, model_names_eff)):
    if name == "SSP-MFN":
        ax.annotate(f'{p:,} params',
                    (p, r2), textcoords="offset points",
                    xytext=(10, -3), fontsize=8.5, fontweight='bold',
                    color=C_OURS, va='center', ha='left')
    elif name == "MulT":
        ax.annotate(f'{p:,}',
                    (p, r2), textcoords="offset points",
                    xytext=(-8, 0), fontsize=7.5, color='#555555',
                    va='center', ha='right')
    elif name == "LMF":
        ax.annotate(f'{p:,}',
                    (p, r2), textcoords="offset points",
                    xytext=(-3, -14), fontsize=7.5, color=C_SOTA,
                    fontweight='bold', va='top', ha='right')
    elif name == "TFN":
        ax.annotate(f'{p:,}',
                    (p, r2), textcoords="offset points",
                    xytext=(10, 0), fontsize=7.5, color='#555555',
                    va='center', ha='left')
    elif name == "MLP+MMD":
        pass
    elif name == "MLP+CORAL":
        ax.annotate(f'10,278',
                    (p, r2), textcoords="offset points",
                    xytext=(10, 0), fontsize=7.5, color='#555555',
                    va='center', ha='left')
    else:
        ax.annotate(f'{name}', (p, r2), textcoords="offset points",
                    xytext=(10, 5), fontsize=7.5, color='gray')

ratio_params = max(params) / params[0]
ratio_r2 = r2_all[0] / max(0.001, max(r2_all[1:]))
ax.annotate(f'{ratio_params:.0f}$\\times$ fewer params\n'
            f'{ratio_r2:.0f}$\\times$ higher $R^2$ vs MulT',
            xy=(params[0], r2_all[0]),
            xytext=(params[0]*5, r2_all[0] - 0.02),
            fontsize=8, color=C_OURS, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.5,
                            connectionstyle='arc3,rad=0.2'))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (log scale)')
ax.set_ylabel('$R^2$')
ax.set_title('Parameter Efficiency: SSP-MFN vs. Multimodal SOTA', fontweight='bold')
ax.set_ylim(-0.04, 0.15)
ax.grid(True, alpha=0.15, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.lines import Line2D
legend_eff = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor=C_OURS,
           markersize=11, label='SSP-MFN (Ours)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=C_SOTA,
           markersize=7, label='LMF (Low-rank Fusion)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ML_LIGHT,
           markersize=7, label='MLP+MMD / MLP+CORAL (DA-enhanced MLP)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ML,
           markersize=7, label='TFN (Tensor Fusion Network)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=C_ML,
           markersize=7, label='MulT (Multimodal Transformer)'),
]
ax.legend(handles=legend_eff, loc='lower center',
          bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7.5,
          framealpha=0.95, edgecolor='#cccccc',
          handletextpad=0.5, labelspacing=0.5,
          columnspacing=1.4, borderpad=0.6)

ax.axhspan(0.08, 0.15, xmin=0, xmax=0.35, alpha=0.04, color=C_OURS)
ax.text(0.03, 0.95, 'Efficient\nregion',
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        color=C_OURS, alpha=0.7, fontstyle='italic')

plt.tight_layout()
plt.savefig(FIG/"FigS3_param_efficiency_v2.png", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.savefig(FIG/"FigS3_param_efficiency_v2.pdf", dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    -> FigS3_param_efficiency_v2.png/pdf")

print(f"\n  All done! Figures saved to: {FIG}/")
print("    Figure1_performance_v2    -- Main Fig.1")
print("    Figure2_mechanism_v2      -- Main Fig.2")
print("    FigS1_gate_ethnic_v2      -- Supplementary")
print("    FigS2_permutation_v2      -- Supplementary")
print("    FigS3_param_efficiency_v2 -- Supplementary")
