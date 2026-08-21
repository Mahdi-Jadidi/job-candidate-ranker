# Job Candidate Ranker

TalentMatch is a reproducible candidate-to-job ranking pipeline.

It joins application, candidate, and job tables, builds a mixed-type feature matrix, trains a gradient-boosted relevance model, writes test scores, and reports grouped NDCG when labels are available.

    pip install -e ".[dev]"
    job-ranker --data-dir . --output-dir artifacts

The canonical implementation is under src/job_candidate_ranker; CI runs independently of any notebook.

