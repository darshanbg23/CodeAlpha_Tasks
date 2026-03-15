import pandas as pd
import numpy as np


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def display_dataset_info(df, dataset_name="Dataset"):
    print(f"\n{'='*60}")
    print(f"{dataset_name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn Names:")
    print(df.columns.tolist())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values")
    else:
        print(missing[missing > 0])
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"{'='*60}\n")


def show_class_distribution(df, target_column):
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in dataset")
        return
    
    print(f"\nClass Distribution for '{target_column}':")
    print(df[target_column].value_counts())
    print(f"\nClass Distribution (%):")
    print(df[target_column].value_counts(normalize=True) * 100)


def handle_missing_values(df, strategy='drop'):
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    else:
        return df