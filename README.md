# SSP-MFN: Social-Skill Prediction via Modality-gated Fusion Network

Reference implementation, experiments, and figure-generation code for the paper
**"Social-Skill Prediction via Modality-gated Fusion Network (SSP-MFN): An Interpretable
Multimodal Framework for Naturalistic Ethnic Music Interaction."**

SSP-MFN is a compact group-conditioned predictor that combines dimension-level modality
routing (a per-scale extension of the Gated Multimodal Unit) with a 1D-specialised
group-conditioned AdaIN normalisation block. It predicts covariate-adjusted change across
six social-skill scales (ICS, IRI, CSAS, SSCS, IOS, SCI-2) from three input streams: audio,
cultural metadata, and participant features.

## Repository structure

```
src/        Model, baselines, and all experiment scripts
features/   Pre-extracted embeddings (MERT audio, XLM-R text) and intermediate features
results/    Per-experiment outputs (JSON) used to build the paper tables and figures
scripts/    Shell helpers for the audio-corpus retrieval pipeline
figures_v2/ Final paper figures (PDF) produced by the plotting scripts
configs/    Placeholder for run configurations
```

Key source files in `src/`:

| File | Role |
|------|------|
| `ssp_mfn.py` | SSP-MFN model: modal projectors, dimension-level gated fusion, cultural AdaIN, six-dimensional regression head |
| `cmaf_net.py` | Cross-modal attention fusion variant used in robustness checks |
| `EXP1_sspmfn_main.py` | Main experiment: 5-fold GroupKFold CV of SSP-MFN against nine baselines |
| `EXP2_ablation_matrix.py`, `EXP1_ablation_10seeds.py`, `EXP15_ablation_bootstrap.py` | Ablations on gating and AdaIN conditioning |
| `EXP12_leave_one_group_out.py`, `EXP18_logo_ablation.py` | Leave-one-group-out (out-of-distribution) evaluation |
| `EXP16_falsification_controls.py`, `EXP17_falsification_primary.py`, `EXP9_permutation_sanity.py` | Falsification controls (shuffled-label, capacity-matched zero-embedding) and permutation tests |
| `EXP11_bootstrap_scale_comparison.py`, `EXP7_bootstrap_ci.py` | Bootstrap confidence intervals and scale comparisons |
| `EXP19_item_total_iri_sci2.py` | Item-total reliability analysis for the two low-reliability instruments |
| `baseline_b1_tabular.py`, `baseline_b2_text.py`, `baseline_b3_audio_transfer.py`, `baselines_multimodal.py`, `baselines_domain_adapt.py` | Linear, text-only, audio-transfer, multimodal (TFN/MulT/LMF), and domain-adaptation baselines |
| `extract_audio_embeddings.py`, `extract_text_embeddings.py` | Feature extraction (MERT audio, XLM-R text) |
| `real_v2_data_preparer.py` | Preparer and validator for the real-collected v2 dataset used in the released experiments |
| `plot_final_figures_v2.py`, `plot_all_figures.py` | Figure generation |
| `RE1`–`RE10` | Supporting real-data analyses (UMAP, choral synchrony, IRI CFA, cross-domain distance, culture statistics) |

## Requirements

Python 3.10+ with:

```
pip install torch torchaudio numpy pandas scipy scikit-learn matplotlib librosa
```

The code runs on CPU and on Apple Silicon (`mps`); device selection is automatic. A GPU is
not required for the released real-data experiments.

## Data

The released experiments run on the **real-collected v2 dataset** (850 session rows), integrated from
field-collected study records and real public/crawled corpora. The v2 tables are validated by
`src/real_v2_data_preparer.py` and consist of `participant_table_v2.csv`,
`session_table_v2.csv`, `audio_metadata_table_v2.csv`, and `scale_table_v2.csv`.
Restricted participant recordings and survey responses from the three Chinese ethnic-minority
communities (Dong, Tibetan, and Mongolian) remain anonymized and are distributed only in the
de-identified tabular form used by the scripts. Public/crawled corpora retain source metadata such as
license, DOI, source URL, and sample-origin fields.

The label is the covariate-adjusted residual change per scale,
`Δy_adj = y_post − (β₀ + β₁·y_pre)`, and cross-validation uses `GroupKFold` over `participant_id`.

## Reproducing the experiments

The scripts use absolute paths anchored at a project root. Before running, set the `ROOT`
path near the top of each script (for example in `EXP1_sspmfn_main.py`) to the directory that
contains the data and `实验/` (experiment) folders on your machine.

Typical order:

```
python src/real_v2_data_preparer.py --data_dir "../数据/数据v2"  # validate real-collected tables
python src/EXP1_sspmfn_main.py          # main CV: SSP-MFN vs. baselines
python src/EXP2_ablation_matrix.py      # gating / AdaIN ablations
python src/EXP12_leave_one_group_out.py # out-of-distribution boundary
python src/EXP16_falsification_controls.py
python src/plot_final_figures_v2.py     # regenerate paper figures
```

Each experiment writes its metrics to `results/` as JSON; the plotting scripts read those
files to rebuild the figures in `figures_v2/`.

## Notes

- `results/` and `features/` ship with pre-computed outputs so the tables and figures can be
  inspected without rerunning the full pipeline.
- The hard-coded `ROOT` paths reflect the original development machine. Editing them to your
  local layout is the only change needed to rerun the real-data experiments.

## Citation

If you use this code, please cite the SSP-MFN paper. A BibTeX entry will be added once the
article is published.
