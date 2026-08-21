# Job Candidate Ranker

TalentMatch is a learning-to-rank project for prioritizing candidate applications within each job opening. It is built around the ranking question recruiters actually face: who should be reviewed first for this specific role?

## Problem

Application datasets mix candidate history, skills, salary expectations, job requirements, and job-level competition. A global relevance score is not enough; predictions must rank candidates correctly inside each job pool.

## What was achieved

The original feature-rich ranking experiment produced **mean NDCG@10 of 0.8585** in GroupKFold validation and **0.6822 MAP@5**. A time-oriented hold-out reached **NDCG@10 of 0.8753** and **MAP@5 of 0.6381**. These results were obtained from candidate-job matching features, job context, and an ensemble of gradient-boosted models.

## Current pipeline

- Merges labelled and unlabelled applications with candidate and job tables.
- Handles mixed numeric/categorical fields through a reproducible preprocessing pipeline.
- Fits a gradient-boosted relevance regressor and scores the test application set.
- Writes predictions, model artifact, training metadata, and grouped NDCG where labels are available.

## Reproduce

```bash
pip install -e ".[dev]"
job-ranker --data-dir . --output-dir artifacts
```

Required data files are `applications_train.csv`, `applications_test.csv`, `candidates.csv`, and `jobs.csv`. The executable source is under `src/job_candidate_ranker` and CI tests the ranking metric implementation.
