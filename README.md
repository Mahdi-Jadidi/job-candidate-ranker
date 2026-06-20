# TalentMatch — Learning to Rank for Job-Candidate Matching

This project builds a learning-to-rank pipeline for matching candidates to job postings. It ranks job-candidate applications by predicted relevance using engineered matching features, job-level context, out-of-fold target encoding, and an ensemble of gradient-boosted models.

## What the project does

- Loads applications, candidates, and jobs datasets.
- Performs exploratory data analysis for labels, applications per job, candidate profiles, and job attributes.
- Cleans and normalizes salary, education, English proficiency, seniority, remote policy, and company size.
- Extracts skill/category overlap between candidates and jobs.
- Engineers 35+ candidate-job matching features.
- Adds job-level contextual features such as within-job rank percentiles.
- Uses out-of-fold target encoding for `industry`, `current_title`, and `job_title`.
- Trains a ranking ensemble:
  - `HistGradientBoostingRegressor`
  - `HistGradientBoostingClassifier`
- Evaluates with NDCG@10 and MAP@5.
- Generates a submission file for test applications.

## Dataset

The project uses four CSV files:

- `applications_train.csv` — 118,772 labeled applications
- `applications_test.csv` — 52,700 test applications
- `candidates.csv` — 50,000 candidate profiles
- `jobs.csv` — 5,000 job postings

Target:

- `relevance_label` — relevance score for a candidate-job application

## How to run

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Then run:

```text
Learning to Rank.ipynb
```

## Feature engineering

The notebook creates a 53-feature matrix from:

- Candidate profile features
- Job requirement features
- Skill overlap ratio and absolute overlap
- Missing and extra skills
- Skill-category overlap
- Experience gap and in-range experience
- Salary mismatch and affordability
- Location match and relocation interaction
- Seniority gap and seniority match
- Job-level context:
  - skill overlap rank percentile
  - experience rank percentile
  - salary rank percentile
  - above-pool-median skills
  - pool size
- Out-of-fold target encoding for categorical fields

## Evaluation metrics

### NDCG@10

Measures whether highly relevant candidates are ranked near the top of each job's candidate list.

### MAP@5

Measures average precision over the top 5 ranked candidates, treating relevance labels of 3 or higher as relevant.

## Results

### GroupKFold cross-validation

| Metric | Mean | Std |
|---|---:|---:|
| NDCG@10 | **0.8585** | 0.0025 |
| MAP@5 | **0.6822** | 0.0061 |

### Hold-out evaluation

The last two months of training data were used as a hold-out set.

| Metric | Value |
|---|---:|
| Hold-out NDCG@10 | **0.8753** |
| Hold-out MAP@5 | **0.6381** |

### Final submission

- Test applications scored: **52,700**
- Final feature matrix: **53 features**
- Ensemble blend: `0.6 * regressor_score + 0.4 * classifier_expected_relevance`

## Key findings

- Skill overlap is one of the strongest candidate-job matching signals.
- Job-level context helps compare candidates within the same job pool.
- Out-of-fold target encoding reduces leakage when encoding categorical fields.
- The ensemble improves ranking stability compared with a single model.

## Suggested improvements

- Add semantic text matching for job titles, candidate titles, and skills.
- Add time-aware validation splits if production data has strong temporal drift.
- Calibrate classifier probabilities before blending.
- Compare with LambdaMART / XGBoost ranking objectives.

## Files

- `Learning to Rank.ipynb` — complete ranking pipeline
- `applications_train.csv` — labeled training applications
- `applications_test.csv` — test applications
- `candidates.csv` — candidate profiles
- `jobs.csv` — job postings
- `Project Description.pdf` — project brief
