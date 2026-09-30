# Sounding Fair: Auditing Musical-Tradition Bias and the Limits of Disentanglement

This repository contains the full experimental pipeline for the paper
*"Sounding Fair: Auditing Musical-Tradition Bias and the Limits of
Disentanglement"*, submitted to IEEE Transactions on Artificial
Intelligence as a Brief Manuscript.

---

## 1. Overview

The pipeline audits whether a self-supervised, content-based music
retrieval embedding preserves the internal musicological structure of
four underrepresented musical traditions (Carnatic, Hindustani,
Jingju, Arab-Andalusian), and tests whether a disentanglement-based
mitigation can correct any bias found. All programs are numbered in
the order they were run and are designed to be executed sequentially
in a Google Colab environment, each producing cached artifacts
consumed by later stages.

## 2. Pipeline Stages

| Stage | Programs | Purpose |
|---|---|---|
| E0 — Data Extraction | 01–07 | Stage the dominant pool (FMA-small, MTG-Jamendo subset) and four tradition corpora from source archives into clean, verified per-track manifests. |
| E1 — Backbone Training | 08 | Train a SimCLR-style contrastive CNN encoder on the pooled 16,976-track corpus. No tradition or group label is used in training. |
| E2 / E2b — Feature-Space Probes | 09–10 | Probe whether tonic-normalized pitch-class features carry more tradition-specific signal than standard chroma; a follow-up compares against a real-tonic cent-scale filterbank feature representation from prior published work (Program 10). |
| E3 — Sub-band Leakage Probe | 11 | Determine whether a loudness confound detected in intermediate encoder activations is architectural or training-induced, to decide how the confound should be targeted downstream. |
| E4 — Bias Audit | 12–13 | Stage four held-out, musicologically-labeled fidelity corpora (raga, nawba, role-type) and compute the paper's four core audit metrics: self-consistency, exposure disparity, cross-tradition retrievability, and intra-group fidelity. |
| E5 — Baseline Comparison | 14–16 | Four baselines against the vanilla backbone: post-hoc fair re-ranking, signal-level loudness/EQ normalization, an HSIC penalty on raw features with no learned encoder, and an HSIC penalty on a fully retrained encoder. |
| E6 — Mitigation | 18–19 | Train a disentangled VAE with auxiliary content/style supervision, generate counterfactual embeddings via distance-quantile style transfer, and train a fairness-regularized residual refinement network. |
| E6 (cont.) — Sweep, Diversity, Ablation | 20–23 | A penalty-weight sweep across two orders of magnitude, a finer follow-up sweep with a corrected reproducibility fix, a retrieval-diversity/homogenization analysis, and an ablation testing whether the auxiliary semantic supervision was necessary. |

NOTE: There is no program '17'.

## 3. Datasets

All datasets used are third-party, publicly documented research
corpora:

- **FMA-small** and an 8,000-track subset of **MTG-Jamendo** (dominant pool)
- **Saraga Carnatic** and **Saraga Hindustani** (Carnatic and Hindustani traditions)
- **Jingju base corpus** and the **JaCRC supplement** (Jingju tradition)
- **Arab-Andalusian dataset** (Arab-Andalusian tradition)
- A raga-recognition dataset providing held-out Carnatic and Hindustani raga labels used only for the Intra-Group Fidelity metric, never for training

No dataset is redistributed in this repository. Each extraction
program (01–07) documents its expected source archive layout; users
must obtain each dataset from its original distributor.

## 4. Reproducibility Notes

- **Storage layer.** Every artifact this pipeline produces — trained
  models, cached embeddings, computed features, result JSON files,
  and figures — is persisted to a single Google Drive directory,
  `PROJECT_DIR`, whose path is set as a constant near the top of each
  script (currently `/content/drive/MyDrive/paper/sounding-fair/`).
  There is no separate cache or mirror directory; every script reads
  its inputs from and writes its outputs to this same location, so
  one path constant is all that needs to be reconfigured to reproduce
  the pipeline under your own Google Drive account.
- **Random seeds.** All stochastic operations (data splits, batch
  sampling, model initialization, bootstrap resampling) are seeded
  (`RANDOM_SEED=42` throughout) for reproducibility, with one
  documented exception: an early version of Program 20/21's sweep did
  not reset the model-initialization seed independently per sweep
  point, which Program 21 corrects.
- **Resumability.** Long-running training scripts (08, 16, 18, 19,
  20, 21, 23) checkpoint after every epoch and resume automatically if
  interrupted.
- **Version history.** Several scripts (notably 10, 16, 18, 19)
  include an in-file change log documenting bugs found and fixed
  during development, including two documented failure modes in the
  fairness-regularized refinement network's loss normalization
  (Program 19) that are directly relevant to interpreting the paper's
  reported results.

## 5. Environment

Developed and run in Google Colab. Core dependencies across the
pipeline: `torch`, `torchaudio`, `librosa`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `tqdm`. Individual scripts list their
exact dependency subset in their header.

## 6. Status

This repository accompanies a manuscript under double-blind review.
Author identity, institutional affiliation, and full citation
information will be added upon acceptance.
