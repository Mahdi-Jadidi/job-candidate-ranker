import numpy as np


def ndcg_at_k(labels, scores, k=10):
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores)[::-1][:k]
    ranked = labels[order]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = np.sum((2**ranked - 1) / discounts)
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg else 0.0


def mean_ndcg(frame, scores, group="job_id"):
    data = frame.assign(_score=scores)
    return float(np.mean([ndcg_at_k(g.relevance_label, g._score) for _, g in data.groupby(group)]))
