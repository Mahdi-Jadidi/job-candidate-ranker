"""
Feature engineering module for the TalentMatch Learning to Rank pipeline.
Contains functions to create skill dictionaries, build base feature vectors,
and compute job-level contextual features.
"""
import pandas as pd
import numpy as np
import re
from . import config

def create_skill_dicts(cands_df, jobs_df):
    """
    Create dictionaries mapping candidate/job IDs to their skill sets and skill categories.

    Parameters
    ----------
    cands_df : pandas.DataFrame
        Processed candidates DataFrame with a 'skills' column.
    jobs_df : pandas.DataFrame
        Processed jobs DataFrame with a 'required_skills' column.

    Returns
    -------
    tuple of (cand_skills_dict, job_skills_dict, cand_cats_dict, job_cats_dict, cand_certs_dict)
        Dictionaries for skills and categories.
    """
    def parse_skills(s):
        if pd.isna(s):
            return set()
        return set(x.strip().lower() for x in str(s).split('|') if x.strip())

    def get_skill_categories(skill_str):
        if pd.isna(skill_str):
            return set()
        skills = set(x.strip().lower() for x in str(skill_str).split('|'))
        cats = set()
        for cat, cat_skills in config.SKILL_CATEGORIES.items():
            if skills & cat_skills:
                cats.add(cat)
        return cats

    cand_skills_dict = {r.candidate_id: parse_skills(r.skills) for r in cands_df.itertuples(index=False)}
    job_skills_dict = {r.job_id: parse_skills(r.required_skills) for r in jobs_df.itertuples(index=False)}

    cand_cats_dict = {r.candidate_id: get_skill_categories(r.skills) for r in cands_df.itertuples(index=False)}
    job_cats_dict = {r.job_id: get_skill_categories(r.required_skills) for r in jobs_df.itertuples(index=False)}

    cand_certs_dict = {r.candidate_id: parse_skills(r.certifications) for r in cands_df.itertuples(index=False)}

    return cand_skills_dict, job_skills_dict, cand_cats_dict, job_cats_dict, cand_certs_dict

def build_features(df, cands_df, jobs_df, skill_dicts=None):
    """
    Build the base feature matrix (35+ features) for a given dataframe of job-candidate pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least 'job_id' and 'candidate_id' columns (e.g., applications train/test).
    cands_df : pandas.DataFrame
        Processed candidates DataFrame.
    jobs_df : pandas.DataFrame
        Processed jobs DataFrame.
    skill_dicts : tuple, optional
        Precomputed tuple (cand_skills_dict, job_skills_dict, cand_cats_dict, job_cats_dict, cand_certs_dict).
        If None, it will be computed inside the function.

    Returns
    -------
    pandas.DataFrame
        Feature matrix with one row per row in df and 45+ columns.
    """
    if skill_dicts is None:
        skill_dicts = create_skill_dicts(cands_df, jobs_df)
    cand_skills_dict, job_skills_dict, cand_cats_dict, job_cats_dict, cand_certs_dict = skill_dicts

    # Global medians for imputation (computed from the candidates data)
    global_exp_median = cands_df['years_experience'].median()
    global_sal_median = cands_df['expected_salary'].median()

    # Columns to merge from candidates and jobs
    candidate_cols = [
        'candidate_id', 'years_experience', 'education_num', 'english_num',
        'expected_salary', 'relocate_num', 'profile_completeness', 'cand_loc',
        'cand_seniority', 'num_certs', 'num_prev_companies'
    ]
    job_cols = [
        'job_id', 'min_years_experience', 'max_years_experience',
        'salary_min', 'salary_max', 'company_size_num', 'remote_num',
        'job_loc', 'job_seniority'
    ]

    # Merge to get features for each pair
    m = df[['job_id', 'candidate_id']].copy()
    m = m.merge(cands_df[candidate_cols], on='candidate_id', how='left')
    m = m.merge(jobs_df[job_cols], on='job_id', how='left')

    # Initialize feature DataFrame
    f = pd.DataFrame(index=m.index)

    # Direct copies with imputation
    f['years_exp'] = m['years_experience'].fillna(general_exp_median)
    f['edu_num'] = m['education_num'].fillna(2)
    f['eng_num'] = m['english_num'].fillna(3)
    f['profile_comp'] = m['profile_completeness'].fillna(50)
    f['relocate'] = m['relocate_num'].fillna(0)
    f['cand_seniority'] = m['cand_seniority'].fillna(2)
    f['num_certs'] = m['num_certs'].fillna(0)
    f['num_prev_companies'] = m['num_prev_companies'].fillna(0)

    # Job experience range
    jmin = m['min_years_experience'].fillna(0)
    jmax = m['max_years_experience'].fillna(m['max_years_experience'].median())
    jmid = (jmin + jmax) / 2

    f['job_min_exp'] = jmin
    f['job_max_exp'] = jmax

    # Salary midpoint
    sal_mid = ((m['salary_min'] + m['salary_max']) / 2).fillna(80_000)
    f['sal_mid'] = sal_mid
    f['company_size'] = m['company_size_num'].fillna(3)
    f['remote'] = m['remote_num'].fillna(0)
    f['job_seniority'] = m['job_seniority'].fillna(2)

    n = len(m)
    # Initialize arrays for skill-based features
    sor = np.zeros(n)          # skill overlap ratio
    soa = np.zeros(n)          # skill overlap absolute
    msk = np.zeros(n)          # missing skills
    ext = np.zeros(n)          # extra skills
    scat = np.zeros(n)         # skill category overlap
    scat_r = np.zeros(n)       # skill category ratio
    tot_c = np.zeros(n)        # total skills candidate
    tot_j = np.zeros(n)        # total skills job
    job_ids = m['job_id'].values
    cand_ids = m['candidate_id'].values

    for i, (jid, cid) in enumerate(zip(job_ids, cand_ids)):
        js = job_skills_dict.get(jid, set())
        cs = cand_skills_dict.get(cid, set())
        jc_cats = job_cats_dict.get(jid, set())
        cc_cats = cand_cats_dict.get(cid, set())

        tot_c[i] = len(cs)
        tot_j[i] = len(js)

        if js:
            ov = len(cs & js)
            soa[i] = ov
            sor[i] = ov / len(js)
            msk[i] = len(js - cs)

        ext[i] = len(cs - js) if js else len(cs)

        if jc_cats:
            cat_ov = len(cc_cats & jc_cats)
            scat[i] = cat_ov
            scat_r[i] = cat_ov / len(jc_cats)

    f['skill_overlap_ratio'] = sor
    f['skill_overlap_abs'] = soa
    f['missing_skills'] = msk
    f['extra_skills'] = ext
    f['skill_cat_overlap'] = scat
    f['skill_cat_ratio'] = scat_r
    f['total_skills_cand'] = tot_c
    f['total_skills_job'] = tot_j
    f['skills_density'] = tot_c / (tot_j + 1)

    # Experience-derived features
    ce = f['years_exp']
    f['exp_gap_mid'] = ce - jmid
    f['exp_below_min'] = np.maximum(0, jmin - ce)
    f['exp_above_max'] = np.maximum(0, ce - jmax)
    f['exp_in_range'] = ((ce >= jmin) & (ce <= jmid)).astype(int)  # Wait, jmid is average? In notebook they used jmin and jmax for in_range.
    # Let's check the notebook: they defined jmid = (jmin + jmax)/2, then exp_in_range = ((ce >= jmin) & (ce <= jmax)).astype(int)
    # I made a mistake above. Let's fix.
    # We'll recalc: we already have jmin and jmax.
    f['exp_in_range'] = ((ce >= jmin) & (ce <= jmax)).astype(int)
    f['exp_squared'] = ce ** 2

    # Salary mismatch and affordability
    csal = m['expected_salary'].fillna(global_sal_median)
    f['sal_mismatch'] = csal - sal_mid
    f['sal_affordable'] = ((csal >= m['salary_min'].fillna(0)) &
                           (csal <= m['salary_max'].fillna(999_999))).astype(int)
    f['sal_ratio'] = (csal / sal_mid.replace(0, np.nan)).fillna(1).clip(0.1, 10)
    f['sal_log_ratio'] = np.log(f['sal_ratio'].clip(0.01, 100))
    f['sal_rel_market'] = (csal / global_sal_median).clip(0.1, 20)

    # Education gap: map job min years to education level
    jed = jmin.map(lambda x: 1 if x < 1 else (2 if x < 3 else 3))
    f['edu_gap'] = f['edu_num'] - jed

    # Location match
    cloc = m['cand_loc']
    jloc = m['job_loc']
    remote_ok = m['remote_num'].fillna(0) >= 1
    lm = np.zeros(n)
    for i in range(n):
        cl = cloc.iloc[i]
        jl = jloc.iloc[i]
        if cl == jl:
            lm[i] = 2
        elif remote_ok.iloc[i] or cl == 'remote':
            lm[i] = 1
    f['loc_match'] = lm

    # Seniority features
    cs_arr = f['cand_seniority'].values
    js_arr = f['job_seniority'].values
    f['seniority_gap'] = cs_arr - js_arr
    f['seniority_gap_abs'] = np.abs(cs_arr - js_arr)
    f['seniority_match'] = (cs_arr == js_arr).astype(int)

    # Interaction features
    f['reloc_x_mismatch'] = f['relocate'] * (lm == 0).astype(int)
    f['experience_x_overlap'] = f['years_exp'] * f['skill_overlap_ratio']
    f['overlap_x_inrange'] = f['skill_overlap_ratio'] * f['exp_in_range']
    f['edu_x_overlap'] = f['edu_num'] * f['skill_overlap_ratio']
    f['eng_x_seniority'] = f['eng_num'] * f['cand_seniority']
    f['profile_x_overlap'] = f['profile_comp'] * f['skill_overlap_ratio']
    f['cert_count_x_overlap'] = f['num_certs'] * f['skill_overlap_ratio']

    return f

def build_job_context_features(df, feature_df):
    """
    Compute job-level contextual features (rank percentages, pool size, etc.).

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe with 'job_id' column (same index as feature_df).
    feature_df : pandas.DataFrame
        Feature matrix (output of build_features) with columns:
        'skill_overlap_ratio', 'years_exp', 'sal_ratio'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['skill_overlap_rank_pct', 'exp_rank_pct', 'sal_rank_pct',
         'above_pool_median_skills', 'pool_size']
    """
    ctx = pd.DataFrame(index=df.index)
    ctx['job_id'] = df['job_id'].values
    ctx['skill_overlap_ratio'] = feature_df['skill_overlap_ratio'].values
    ctx['years_exp'] = feature_df['years_exp'].values
    ctx['sal_ratio'] = feature_df['sal_ratio'].values

    ctx['skill_overlap_rank_pct'] = ctx.groupby('job_id')['skill_overlap_ratio'].rank(pct=True)
    ctx['exp_rank_pct'] = ctx.groupby('job_id')['years_exp'].rank(pct=True)
    ctx['sal_rank_pct'] = ctx.groupby('job_id')['sal_ratio'].rank(pct=True)

    job_med_overlap = ctx.groupby('job_id')['skill_overlap_ratio'].transform('median')
    ctx['above_pool_median_skills'] = (ctx['skill_overlap_ratio'] >= job_med_overlap).astype(int)

    ctx['pool_size'] = ctx.groupby('job_id')['job_id'].transform('count')

    return ctx[['skill_overlap_rank_pct', 'exp_rank_pct', 'sal_rank_pct',
                'above_pool_median_skills', 'pool_size']]