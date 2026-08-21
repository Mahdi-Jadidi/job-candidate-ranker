from job_candidate_ranker.metrics import ndcg_at_k


def test_perfect_ranking_has_unit_ndcg():
    assert ndcg_at_k([3, 2, 0], [0.9, 0.5, 0.1]) == 1.0
