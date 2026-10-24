# StableOmics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)


**A CPU-first, reproducible benchmark of ten cross-cohort transcriptomic ranking methods on four independent bulk colorectal cancer cohorts.**

> **Headline result:** The composite stability score `R_g = S_g · D_g · C_g` is **empirically falsified** for reproducible bulk biomarker discovery. **Cross-cohort rank aggregation** (fixed-effect meta-analysis, SumRank-bulk) matches or exceeds single-cohort differential expression. Direction replication saturates at ceiling on strong-signal diseases and cannot discriminate methods — **top-K overlap at K = 20** should be the primary endpoint.

---

## Overview

Differential-expression (DE) significance in a discovery cohort is not equivalent to reproducible cross-cohort biomarker evidence. This repository contains the full, reproducible pipeline used to benchmark **ten ranking methods** under a **nested leave-one-cohort-out (LOCO)** protocol on four independent bulk CRC cohorts from NCBI GEO.

**Cohorts** — 225 primary CRC cases vs 226 non-tumour mucosa controls, harmonised to **7,542 HGNC genes**:

| Cohort   | Platform | Cases | Controls | Quality imbalance (p) |
|----------|----------|-------|----------|------------------------|
| GSE44076 | GPL13667 | 98    | 148      | **< 0.001**            |
| GSE44861 | GPL3921  | 74    | 37       | 0.067                  |
| GSE20916 | GPL570   | 36    | 24       | 0.294                  |
| GSE32323 | GPL570   | 17    | 17       | 0.757                  |

GSE44076 shows near-total confounding between case-control status and sample quality and is excluded in a prespecified three-cohort sensitivity analysis.

**Methods compared** (10 + 1 ablation):
`Single_DE` · `S_g_only` · `BootstrapSeq` · `SNR_g_only` · `SumRank_Bulk` · `Meta_Effect` · `Ablation_No_Cg (S_g·D_g)` · `StableOmics_WeightedSum` · **`StableOmics_Rg = S_g·D_g·C_g`** (prespecified) · `StableOmics_Rg_SNR` · `Ablation_No_Dg (S_g·C_g)`

---

## Key findings

- **Primary endpoint saturated**: directional replication (DR@K=50) averaged **0.999** across all 40 method × held-out-cohort comparisons. It cannot discriminate methods on strong-signal diseases.
- **Top-K overlap does discriminate** (K = 50, 4 cohorts):

  | Method                       | O_K   |
  |------------------------------|-------|
  | Meta-analysis (fixed effect) | **0.285** |
  | SumRank (bulk)               | **0.280** |
  | Single-cohort DE             | 0.250 |
  | Ablation S_g·D_g             | 0.195 |
  | Composite R_g                | 0.185 |
  | S_g alone                    | 0.175 |
  | SNR                          | 0.085 |
  | BootstrapSeq                 | 0.045 |

- **Ablation collapses the composite**: D_g and C_g are near ceiling for almost all genes (mean D_g = 0.97, mean C_g = 0.92), so `R_g` reduces to `S_g`, which itself underperforms DE.
- **Rank aggregation wins** and the advantage sharpens in the three-cohort sensitivity analysis: ΔO_K vs DE = **+0.093** (meta-analysis) and **+0.073** (SumRank).
- **Power caveat, prespecified**: with C = 4 cohorts the minimum exact two-sided permutation *p* is 0.125. All comparisons are **descriptive**, not inferential.

See the manuscript for the full narrative.

---

## Requirements

- Python **3.10+**
- ~2 GB free disk space (raw data downloaded on demand)
- CPU only — no GPU required
- Peak resident memory: **0.6 GB**
- End-to-end runtime: **~6 min** (B = 100 quick mode) on a single-core Intel i5 laptop

---

## Installation

The repo ships with a pinned `environment.yml`. Its environment name is on the first line — check it with:
