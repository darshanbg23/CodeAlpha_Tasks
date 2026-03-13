import pandas as pd
from pathlib import Path


def load_german_credit(data_path):
    column_names = [
        'checking_account',
        'duration_months',
        'credit_history',
        'purpose',
        'credit_amount',
        'savings_account',
        'employment_since',
        'installment_rate',
        'personal_status',
        'other_debtors',
        'residence_since',
        'property',
        'age',
        'other_installments',
        'housing',
        'existing_credits',
        'job',
        'dependents',
        'telephone',
        'foreign_worker',
        'target'
    ]
    
    df = pd.read_csv(data_path, sep=' ', header=None, names=column_names)
    
    # Convert target from 1/2 to 1/0 (1=good, 0=bad)
    df['target'] = (df['target'] == 1).astype(int)
    
    return df


def explore_dataset(df):
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nTarget proportion:\n{df['target'].value_counts(normalize=True)}")