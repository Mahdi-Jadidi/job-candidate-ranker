"""
Configuration constants for the TalentMatch Learning to Rank pipeline.
"""
import numpy as np

# Random seed for reproducibility
SEED = 42

# Currency conversion rates
IRR_TO_USD = 1 / 42_000
EUR_TO_USD = 1.10

# Seniority extraction patterns
SENIORITY_PATTERNS = [
    (r'\bintern\b|\btrainee\b', 0),
    (r'\bjunior\b|\bjr\.?\b', 1),
    (r'\bmid\b|\bstaff\b', 2),
    (r'\bsenior\b|\bsr\.?\b', 3),
    (r'\blead\b|\bprincipal\b', 4),
    (r'\bstaff\b', 4),
    (r'\bmanager\b|\bdirector\b|\bhead\b|\bvp\b|\bcto\b|\bceo\b', 5),
]

# Education and English proficiency mappings
EDU_MAP = {
    'high school': 1, 'hs': 1, 'secondary': 1,
    'bachelor': 2, 'b.a.': 2, 'b.s.': 2, 'bsc': 2,
    'master': 3, 'm.s.': 3, 'm.a.': 3, 'msc': 3,
    'phd': 4, 'ph.d.': 4, 'doctorate': 4,
}

ENG_MAP = {
    'a1': 1, 'beginner': 1,
    'a2': 2, 'elementary': 2,
    'b1': 3, 'intermediate': 3,
    'b2': 4, 'upper-intermediate': 4,
    'c1': 5, 'advanced': 5,
    'c2': 6, 'fluent': 6, 'bilingual': 6, 'native': 6,
}

# Skill categories for grouping
SKILL_CATEGORIES = {
    'cloud': {'aws-csa', 'aws-sap', 'aws-dva', 'gcp-ace', 'gcp-pca', 'azure-az104', 'azure-az305',
              'azure-az900', 'aws', 'gcp', 'azure', 'cloud', 'terraform', 'ansible', 'pulumi'},
    'data': {'python', 'r', 'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'cassandra', 'dbt',
             'spark', 'hadoop', 'kafka', 'tableau', 'powerbi', 'snowflake', 'databricks',
             'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'},
    'ml': {'tensorflow', 'pytorch', 'scikit-learn', 'ml', 'machine learning', 'deep learning',
           'nlp', 'computer vision', 'keras', 'xgboost', 'lightgbm'},
    'frontend': {'react', 'vue', 'angular', 'svelte', 'html', 'css', 'sass', 'scss', 'typescript',
                 'javascript', 'next.js', 'nuxt', 'redux', 'webpack', 'tailwind'},
    'backend': {'python', 'java', 'node.js', 'spring', 'django', 'fastapi', 'flask', 'go', 'rust',
                'c#', 'php', 'laravel', 'express', 'rails', 'rest api', 'graphql', 'grpc'},
    'devops': {'docker', 'kubernetes', 'jenkins', 'gitlab-ci', 'github actions', 'cicd', 'ci/cd',
               'terraform', 'ansible', 'nginx', 'linux', 'bash', 'prometheus', 'grafana'},
    'mobile': {'swift', 'kotlin', 'flutter', 'react native', 'ios', 'android', 'mobile'},
    'security': {'ceh', 'cissp', 'pci-dss', 'hipaa', 'soc2', 'security', 'penetration testing', 'oauth'},
    'pm': {'scrum', 'agile', 'pmp', 'jira', 'kanban', 'project management', 'scrum-master'},
}

# Global medians for imputation (will be computed from data)
GLOBAL_EXP_MEDIAN = None
GLOBAL_SAL_MEDIAN = None