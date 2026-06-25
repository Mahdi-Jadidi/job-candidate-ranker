"""
Encoding module for the TalentMatch Learning to Rank pipeline.
Contains out-of-fold target encoding implementation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def oof_target_encode(train_series, y, groups, n_splits=5):
    """
    Performs out-of-fold (OOF) target encoding using GroupKFold.

    Parameters
    ----------
    train_series : pandas.Series
        Series of categorical values to encode (e.g., industry, current_title, job_title).
    y : array-like
        Target values (relevance_label).
    groups : array-like
        Group labels for GroupKFold (typically job_id to avoid leakage).
    n_splits : int, default 5
        Number of folds for GroupKFold.

    Returns
    -------
    tuple of (encoded_array, full_mapping, global_mean)
        encoded_array : np.ndarray of shape (len(train_series),) with OOF encoded values.
        full_mapping : dict mapping each category to its global mean target.
        global_mean : float, the mean of y over the entire training set.
    """
    gkf = GroupKFold(n_splits=n_splits)
    encoded = np.zeros(len(train_series))
    gm = float(np.mean(y))

    for tr_idx, va_idx in gkf.split(train_series, y, groups):
        # Compute mapping from training fold
        fold_map = (
            pd.Series(y[tr_idx], index=train_series.iloc[tr_idx].values)
            .groupby(level=0).mean()
        )
        # Apply to validation fold
        encoded[va_idx] = train_series.iloc[va_idx].map(fold_map).fillna(gm).values

    # Compute global mapping (full training)
    full_map = pd.Series(y, index=train_series.values).groupby(level=0).mean()

    return encoded, full_map.to_dict(), gm