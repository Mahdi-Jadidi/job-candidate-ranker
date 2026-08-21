from pathlib import Path
import pandas as pd


def load_data(data_dir: str):
    root = Path(data_dir)
    return tuple(pd.read_csv(root / name) for name in ("applications_train.csv", "applications_test.csv", "candidates.csv", "jobs.csv"))


def join_features(applications, candidates, jobs):
    candidate_key = next((c for c in ("candidate_id", "candidateId", "id") if c in applications and c in candidates), None)
    job_key = next((c for c in ("job_id", "jobId", "id") if c in applications and c in jobs), None)
    result = applications.copy()
    if candidate_key:
        result = result.merge(candidates, on=candidate_key, how="left", suffixes=("", "_candidate"))
    if job_key:
        result = result.merge(jobs, on=job_key, how="left", suffixes=("", "_job"))
    return result
