<div align="center">

# TalentMatch: Job Candidate Ranker

**Learning relevance scores that rank the right candidates within each job pool**

[![CI](https://github.com/Mahdi-Jadidi/job-candidate-ranker/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahdi-Jadidi/job-candidate-ranker/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ranking](https://img.shields.io/badge/Evaluation-NDCG%4010%20%7C%20MAP%405-0F766E)

</div>

## Overview

TalentMatch ranks applications for each open role by combining application records with candidate profiles and job requirements. Unlike a global classifier, the system is evaluated on whether the most relevant candidates rise to the top *within the correct job*.

## Results

| Validation protocol | NDCG@10 | MAP@5 |
|---|---:|---:|
| GroupKFold cross-validation | 0.8585 +/- 0.0025 | **0.6822 +/- 0.0061** |
| Time-oriented hold-out | **0.8753** | 0.6381 |

The stable GroupKFold scores indicate that ranking quality generalizes across job groups. The time hold-out provides a more deployment-like view and exposes the expected shift in top-five precision.

## Data

| File | Rows | Purpose |
|---|---:|---|
| `applications_train.csv` | 118,772 | Labelled candidate-job interactions |
| `applications_test.csv` | 52,700 | Applications to score |
| `candidates.csv` | 50,000 | Candidate profile attributes |
| `jobs.csv` | 5,000 | Job requirements and context |

The original experiment engineered skill overlap, experience gap, salary alignment, location match, seniority, category matches, and within-job context. The current package provides a reproducible mixed-type ranking baseline and artifact contract.

## Pipeline

```mermaid
flowchart LR
    A[Applications] --> D[Candidate-job join]
    B[Candidates] --> D
    C[Jobs] --> D
    D --> E[Numeric and categorical preprocessing]
    E --> F[Gradient-boosted relevance model]
    F --> G[Scores grouped by job]
    G --> H[NDCG@10 evaluation]
    G --> I[predictions.csv]
```

## Why NDCG

NDCG rewards placing highly relevant candidates near the top while allowing graded relevance labels. MAP@5 complements it with a strict view of relevant candidates in the first five review positions.

## Quick start

```bash
git clone https://github.com/Mahdi-Jadidi/job-candidate-ranker.git
cd job-candidate-ranker
pip install -e ".[dev]"
job-ranker --data-dir . --output-dir artifacts
```

## Artifacts

- `model.joblib`: preprocessing plus fitted relevance model.
- `predictions.csv`: one score per test application.
- `metrics.json`: row counts and grouped training NDCG where available.

## Limitations and responsible use

Historical relevance labels can encode hiring bias. The model must not be used as an autonomous hiring decision-maker. A production system requires subgroup fairness analysis, explainability, human review, data-retention controls, and monitoring for changes in job and candidate populations.
