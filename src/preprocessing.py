"""
Preprocessing module for the TalentMatch Learning to Rank pipeline.
Handles currency normalization, education/English mapping, seniority extraction,
and creation of processed candidate and job DataFrames.
"""
import pandas as pd
import numpy as np
import re
from . import config

def normalize_currency_col(df, col):
    """
    Normalize a salary column to USD using currency conversion rates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'salary_currency' column and the column to normalize.
    col : str
        Name of the column to normalize (e.g., 'salary_min' or 'salary_max').

    Returns
    -------
    pandas.DataFrame
        Copy of df with the specified column converted to USD.
    """
    df = df.copy()
    curr = df['salary_currency'].str.strip().str.upper().fillna('USD')
    df[col] = df[col].astype(float)

    df.loc[curr == 'IRR', col] *= config.IRR_TO_USD
    df.loc[curr == 'EUR', col] *= config.EUR_TO_USD

    return df

def extract_seniority(title):
    """
    Extract seniority level from a job title.

    Parameters
    ----------
    title : str
        Job title string.

    Returns
    -------
    int
        Seniority level from 0 (intern/trainee) to 5 (executive).
        Defaults to 2 (mid-level) if title is NaN or no pattern matches.
    """
    if pd.isna(title):
        return 2

    t = str(title).lower()
    for pat, level in reversed(config.SENIORITY_PATTERNS):
        if re.search(pat, t):
            return level

    return 2

def preprocess_data(candidates, jobs):
    """
    Preprocess candidates and jobs DataFrames.

    Steps:
    - Normalize salary columns in jobs to USD.
    - Map education levels to numeric values.
    - Map English proficiency to numeric values.
    - Convert willing_to_relocate to binary.
    - Map company size and remote_allowed to numeric.
    - Create lowercase location columns.
    - Extract seniority from current_title and job_title.
    - Count certifications and previous companies.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Raw candidates DataFrame.
    jobs : pandas.DataFrame
        Raw jobs DataFrame.

    Returns
    -------
    tuple of (cands_c, jobs_c)
        Processed candidates and jobs DataFrames with new columns.
    """
    # Normalize job salaries
    jobs_c = normalize_currency_col(jobs, 'salary_min')
    jobs_c = normalize_currency_col(jobs_c, 'salary_max')

    # Process candidates
    cands_c = candidates.copy()

    # Education and English mapping
    cands_c['education_num'] = cands_c['education_level'].map(
        lambda v: config.EDU_MAP.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan
    )

    cands_c['english_num'] = cands_c['english_proficiency'].map(
        lambda v: config.ENG_MAP.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan
    )

    # Relocate, company size, remote
    cands_c['relocate_num'] = cands_c['willing_to_relocate'].map({'Yes': 1, 'No': 0})
    jobs_c['company_size_num'] = jobs_c['company_size'].map({'1-10': 1, '11-50': 2, '51-200': 3, '201-1000': 4, '1000+': 5})
    jobs_c['remote_num'] = jobs_c['remote_allowed'].map({'No': 0, 'Hybrid': 1, 'Yes': 2})

    # Location columns (lowercase, fill missing)
    cands_c['cand_loc'] = cands_c['candidate_location'].fillna('unknown').str.strip().str.lower()
    jobs_c['job_loc'] = jobs_c['job_location'].fillna('unknown').str.strip().str.lower()

    # Seniority
    cands_c['cand_seniority'] = cands_c['current_title'].map(extract_seniority)
    jobs_c['job_seniority'] = jobs_c['job_title'].map(extract_seniority)

    # Count certifications and previous companies
    cands_c['num_certs'] = cands_c['certifications'].apply(
        lambda x: len(str(x).split('|')) if pd.notna(x) else 0
    )

    cands_c['num_prev_companies'] = cands_c['previous_companies'].apply(
        lambda x: len(str(x).split('|')) if pd.notna(x) else 0
    )

    return cands_c, jobs_c