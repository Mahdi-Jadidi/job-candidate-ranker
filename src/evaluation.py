"""
Evaluation module for the TalentMatch Learning to Rank pipeline.
Contains functions to compute ranking metrics: DCG, NDCG, AP@5, and evaluation per job.
"""
import numpy as np

def dcg_at_k(relevances, k=10):
    """
    Compute Discounted Cumulative Gain at k.

    Parameters
    ----------
    relevances : array-like
        Relevance scores (in the order they appear in the list).
    k : int, default 10
        Cutoff rank.

    Returns
    -------
    float
        DCG@k.
    """
    r = np.asarray(relevances[:k], dtype=float)
    if not len(r):
        return 0.0
    return ((2**r - 1) / np.log2(np.arange(2, len(r) + 2))).sum()

def ndcg_at_k(true_labels, pred_scores, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Parameters
    ----------
    true_labels : array-like
        Ground truth relevance labels.
    pred_scores : array-like
        Predicted scores (higher means more relevant).
    k : int, default 10
        Cutoff rank.

    Returns
    -------
    float
        NDCG@k.
    """
    order = np.argsort(pred_scores)[::-1]
    dcg = dcg_at_k(np.array(true_labels)[order], k)
    idcg = dcg_at_k(np.sort(true_labels)[::-1], k)
    return dcg / idcg if idcg > 0 else 0.0

def ap_at_5(true_labels, pred_scores):
    """
    Compute Average Precision at k=5, treating relevance >= 3 as relevant.

    Parameters
    ----------
    true_labels : array-like
        Ground truth relevance labels.
    pred_scores : array-like
        Predicted scores.

    Returns
    -------
    float
        AP@5.
    """
    order = np.argsort(pred_scores)[::-1][:5]
    binary = (np.array(true_labels)[order] >= 3).astype(int)
    n_rel = min(sum(np.array(true_labels) >= 3), 5)

    if n_rel == 0:
        return 0.0

    precisions = []
    hits = 0
    for i, b in enumerate(binary, 1):
        if b:
            hits += 1
            precisions.append(hits / i)

    return sum(precisions) / n_rel if precisions else 0.0

def evaluate_predictions(df_true, scores):
    """
    Compute mean NDCG@10 and mean MAP@5 across all jobs.

    Parameters
    ----------
    df_true : pandas.DataFrame
        DataFrame with columns ['job_id', 'relevance_label'].
    scores : array-like
        Predicted scores aligned with df_true rows.

    Returns
    -------
    tuple of (mean_ndcg, mean_map5)
        Average NDCG@10 and average MAP@5 across jobs.
    """
    df = df_true.copy()
    df['score'] = scores

    ndcgs, aps = [], []
    for jid, grp in df.groupby('job_id'):
        ndcgs.append(ndcg_at_k(grp['relevance_label'].values, grp['score'].values))
        aps.append(ap_at_5(grp['relevance_label'].values, grp['score'].values))

    return np.mean(ndcgs), np.mean(aps)