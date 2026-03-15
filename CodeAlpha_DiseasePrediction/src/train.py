import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import load_dataset, display_dataset_info, show_class_distribution, handle_missing_values


def validate_inputs(file_path, target_column):
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    return df


def get_dataset_name(file_path):
    return os.path.basename(file_path).replace(".csv", "")


def main(file_path, target_column):
    dataset_name = get_dataset_name(file_path)
    
    print("\n" + "="*70)
    print(f"Disease Prediction Model Training - {dataset_name.upper()}")
    print("="*70)
    
    df = validate_inputs(file_path, target_column)
    
    display_dataset_info(df, "Dataset")
    
    df = handle_missing_values(df, strategy='mean')
    show_class_distribution(df, target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    training_columns = list(X.columns)
    
    print("\n" + "="*70)
    print("Train-Test Split (80-20)")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\n" + "="*70)
    print("Building Preprocessing Pipeline")
    print("="*70)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print(f"Numerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    print("\n" + "="*70)
    print("Model Training")
    print("="*70)
    pipeline.fit(X_train, y_train)
    print("LogisticRegression model trained with preprocessing pipeline")
    
    print("\n" + "="*70)
    print("Model Evaluation")
    print("="*70)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix")
    print(cm)
    
    print("\n" + "="*70)
    print("Saving Model and Training Metadata")
    print("="*70)
    os.makedirs('models', exist_ok=True)
    
    model_file = f'models/{dataset_name}_model.pkl'
    columns_file = f'models/{dataset_name}_columns.pkl'
    
    joblib.dump(pipeline, model_file)
    joblib.dump(training_columns, columns_file)
    print(f"Model saved to {model_file}")
    print(f"Training columns saved to {columns_file}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70 + "\n")


def print_usage():
    print("\nUsage: python train.py <dataset_path> <target_column>")
    print("\nExample:")
    print("  python train.py data/heart.csv target")
    print("  python train.py data/diabetes.csv target")
    print("  python train.py data/breast_cancer.csv target")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    file_path = sys.argv[1]
    target_column = sys.argv[2]
    
    main(file_path, target_column)