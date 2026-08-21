from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .data import join_features, load_data
from .metrics import mean_ndcg


def _matrix(frame):
    excluded = {"relevance_label", "job_id", "candidate_id", "application_id"}
    return frame.drop(columns=[c for c in excluded if c in frame], errors="ignore")


def run(data_dir: str, output_dir: str = "artifacts") -> dict:
    train_df, test_df, candidates, jobs = load_data(data_dir)
    train_df = join_features(train_df, candidates, jobs)
    test_df = join_features(test_df, candidates, jobs)
    target = train_df["relevance_label"].astype(float)
    X = _matrix(train_df)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([( "preprocess", prep), ("model", HistGradientBoostingRegressor(max_iter=100, random_state=42))])
    model.fit(X, target)
    scores = model.predict(_matrix(test_df))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    pd.DataFrame({"score": scores}).to_csv(output / "predictions.csv", index=False)
    metrics = {"train_rows": len(train_df), "test_rows": len(test_df)}
    if "job_id" in train_df:
        metrics["train_ndcg_at_10"] = mean_ndcg(train_df, model.predict(X))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
