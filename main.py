"""
Main pipeline script for the TalentMatch Learning to Rank project.

This script orchestrates the entire workflow:
1. Load and preprocess data
2. Engineer features (base, contextual, encoded)
3. Train ranking ensemble (HistGradientBoostingRegressor + HistGradientBoostingClassifier)
4. Evaluate using cross-validation and hold-out (optional)
5. Generate submission file for test applications.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import our custom modules
from src import data_loader
from src import preprocessing
from src import feature_engineering
from src import encoding
from src import evaluation
from src import model

def main():
    print("Loading data...")
    apps_train, apps_test, candidates, jobs = data_loader.load_data()
    print(f"Training applications: {apps_train.shape}")
    print(f"Test applications:     {apps_test.shape}")
    print(f"Candidates:            {candidates.shape}")
    print(f"Jobs:                  {jobs.shape}")

    # Step 1: Preprocess candidates and jobs
    print("\nPreprocessing candidates and jobs...")
    cands_c, jobs_c = preprocessing.preprocess_data(candidates, jobs)
    print("Preprocessing complete.")

    # Step 2: Create skill dictionaries (to be reused)
    print("\nCreating skill dictionaries...")
    skill_dicts = feature_engineering.create_skill_dicts(cands_c, jobs_c)
    print("Skill dictionaries created.")

    # Step 3: Build base features for training and test
    print("\nBuilding base feature matrices...")
    X_train_base = feature_engineering.build_features(apps_train, cands_c, jobs_c, skill_dicts=skill_dicts)
    X_test_base = feature_engineering.build_features(apps_test, cands_c, jobs_c, skill_dicts=skill_dicts)
    print(f"Base features train: {X_train_base.shape}, test: {X_test_base.shape}")

    # Step 4: Build job-level contextual features
    print("\nBuilding job-level contextual features...")
    X_train_ctx = feature_engineering.build_job_context_features(apps_train, X_train_base)
    X_test_ctx = feature_engineering.build_job_context_features(apps_test, X_test_base)
    print(f"Contextual features train: {X_train_ctx.shape}, test: {X_test_ctx.shape}")

    # Step 5: Out-of-fold target encoding for categorical columns
    print("\nApplying out-of-fold target encoding...")
    # We need to encode: industry, current_title, job_title
    # Prepare training metadata (for OOF) and test metadata (same index-aligned with apps_train
    train_meta = (
        apps_train[['candidate_id', 'job_id']]
        .merge(cands_c[['candidate_id', 'current_title']], on='candidate_id', how='left')
        .merge(jobs_c[['job_id', 'industry', 'job_title']], on='job_id', how='left')
    )
    test_meta = (
        apps_test[['candidate_id', 'job_id']]
        .merge(cands_c[['candidate_id', 'current_title']], on='candidate_id', how='left')
        .merge(jobs_c[['job_id', 'industry', 'job_title']], on='job_id', how='left')
    )

    encode_cols = ['industry', 'current_title', 'job_title']
    enc_train = pd.DataFrame(index=apps_train.index)
    enc_test = pd.DataFrame(index=apps_test.index)

    y_train = apps_train['relevance_label'].values
    groups = apps_train['job_id'].values  # Group by job_id to avoid leakage

    for col in encode_cols:
        oof_arr, full_map, gm = encoding.oof_target_encode(
            train_meta[col].fillna('__missing__'),
            y_train,
            groups
        )
        enc_train[f'{col}_enc'] = oof_arr
        # For test, map using the full_map learned from training
        enc_test[f'{col}_enc'] = test_meta[col].fillna('__missing__').map(full_map).fillna(gm).values

    print(f"Encoded features train: {enc_train.shape}, test: {enc_test.shape}")

    # Step 6: Combine all features
    print("\nCombining all features...")
    X_train = pd.concat([
        X_train_base.reset_index(drop=True),
        X_train_ctx.reset_index(drop=True),
        enc_train.reset_index(drop=True)
    ], axis=1)

    X_test = pd.concat([
        X_test_base.reset_index(drop=True),
        X_test_ctx.reset_index(drop=True),
        enc_test.reset_index(drop=True)
    ], axis=1)

    print(f"Final feature matrix train: {X_train.shape}, test: {X_test.shape}")
    print(f"Nulls in train: {X_train.isnull().sum().sum()}, test: {X_test.isnull().sum().sum()}")

    # Optional: Cross-validation evaluation (as in the notebook)
    print("\nRunning GroupKFold cross-validation (5 folds)...")
    from sklearn.model_selection import GroupKFold
    N_SPLITS = 5
    gkf = GroupKFold(n_splits=N_SPLIT)
    groups_cv = apps_train['job_id'].values

    fold_ndcg = []
    fold_map = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_cv)):
        Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        apps_va = apps_train.iloc[va_idx]

        # Train models on this fold
        reg = model.train_regressor(Xtr, ytr)
        clf = model.train_classifier(Xtr, ytr)

        # Predict
        score_reg = reg.predict(Xva)
        proba = clf.predict_proba(Xva)
        classes = clf.classes_.astype(float)
        score_clf = proba @ classes
        score_blend = 0.6 * score_reg + 0.4 * score_clf

        # Evaluate
        val_df = apps_va[['job_id', 'relevance_label']].copy()
        ndcg, map5 = evaluation.evaluate_predictions(val_df, score_blend)

        fold_ndcg.append(ndcg)
        fold_map.append(map5)

        print(f"  Fold {fold + 1}: NDCG@10={ndcg:.4f}  MAP@5={map5:.4f}")

    print(f"\nCV NDCG@10: {np.mean(fold_ndcg):.4f} ± {np.std(fold_ndcg):.4f}")
    print(f"CV MAP@5  : {np.mean(fold_map):.4f} ± {np.std(fold_map):.4f}")

    # Optional: Hold-out evaluation (last 2 months of training data)
    print("\nHold-out evaluation (last 2 months of training data)...")
    cutoff = pd.Timestamp('2024-05-01')
    mask_ho = apps_train['application_date'] >= cutoff
    X_ho_train = X_train[~mask_ho.values]
    X_ho_val = X_train[mask_ho.values]
    y_ho_train = y_train[~mask_ho.values]
    y_ho_val = y_train[mask_ho.values]
    apps_ho_val = apps_train[mask_ho]

    reg_ho = model.train_regressor(X_ho_train, y_ho_train)
    clf_ho = model.train_classifier(X_ho_train, y_ho_train)

    score_reg_ho = reg_ho.predict(X_ho_val)
    proba_ho = clf_ho.predict_proba(X_ho_val)
    classes_ho = clf_ho.classes_.astype(float)
    score_clf_ho = proba_ho @ classes_ho
    score_ho_blend = 0.6 * score_reg_ho + 0.4 * score_clf_ho

    val_df_ho = apps_ho_val[['job_id', 'relevance_label']].copy()
    ndcg_ho, map5_ho = evaluation.evaluate_predictions(val_df_ho, score_ho_blend)
    print(f"Hold-out NDCG@10: {ndcg_ho:.4f}")
    print(f"Hold-out MAP@5  : {map5_ho:.4f}")

    # Step 7: Train final models on full training data
    print("\nTraining final models on full training set...")
    final_reg = model.train_regressor(X_train, y_train)
    final_clf = model.train_classifier(X_train, y_train)
    print("Final models trained.")

    # Step 8: Generate predictions on test data
    print("\nGenerating predictions on test set...")
    test_score_reg = final_reg.predict(X_test)
    test_proba = final_clf.predict_proba(X_test)
    test_classes = final_clf.classes_.astype(float)
    test_score_clf = test_proba @ test_classes
    test_score_blend = 0.6 * test_score_reg + 0.4 * test_score_clf

    print(f"Test score stats: min={test_score_blend.min():.4f}, "
          f"max={test_score_blend.max():.4f}, "
          f"mean={test_score_blend.mean():.4f}")

    # Step 9: Create submission
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'application_id': apps_test['application_id'].values,
        'score': test_score_blend
    })

    # Save to CSV
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

if __name__ == "__main__":
    main()