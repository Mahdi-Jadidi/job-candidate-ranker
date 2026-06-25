"""
Data loading module for the TalentMatch Learning to Rank pipeline.
"""
import pandas as pd

def load_data(
    apps_train_path: str = 'applications_train.csv',
    apps_test_path: str = 'applications_test.csv',
    candidates_path: str = 'candidates.csv',
    jobs_path: str = 'jobs.csv'
):
    """
    Load the four CSV files and parse application dates.

    Parameters
    ----------
    apps_train_path : str, default 'applications_train.csv'
        Path to the training applications CSV.
    apps_test_path : str, default 'applications_test.csv'
        Path to the test applications CSV.
    candidates_path : str, default 'candidates.csv'
        Path to the candidates CSV.
    jobs_path : str, default 'jobs.csv'
        Path to the jobs CSV.

    Returns
    -------
    tuple of (apps_train, apps_test, candidates, jobs)
        DataFrames with application_date parsed as datetime.
    """
    apps_train = pd.read_csv(apps_train_path)
    apps_test = pd.read_csv(apps_test_path)
    candidates = pd.read_csv(candidates_path)
    jobs = pd.read_csv(jobs_path)

    apps_train['application_date'] = pd.to_datetime(apps_train['application_date'])
    apps_test['application_date'] = pd.to_datetime(apps_test['application_date'])

    return apps_train, apps_test, candidates, jobs