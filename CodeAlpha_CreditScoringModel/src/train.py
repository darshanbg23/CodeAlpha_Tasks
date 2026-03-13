import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from data_loader import load_german_credit, explore_dataset
from preprocessing import identify_features, build_preprocessor

def print_metrics(y_true, y_pred, y_pred_prob, model_name):
    print(f"\n{model_name} - Results on Test Set")
    print("=" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }


def main():
    
    # Load and explore data
    print("Loading German Credit Dataset...")
    data_path = Path(__file__).parent.parent / 'data' / 'german.data'
    df = load_german_credit(data_path)
    
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    explore_dataset(df)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    cat_features, num_features = identify_features(df)
    preprocessor = build_preprocessor(cat_features, num_features)
    
    # 80/20 Stratified Train-Test Split
    print("\n" + "="*60)
    print("DATA SPLITTING (80/20 Stratified)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set:     {X_test.shape[0]} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Class distribution in training set:")
    print(f"  Good credit: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"  Bad credit:  {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    
    # Baseline Model: Logistic Regression
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
    print("="*60)
    
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)
    
    lr_test_pred = lr_pipeline.predict(X_test)
    lr_test_pred_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    
    lr_metrics = print_metrics(y_test, lr_test_pred, lr_test_pred_prob, "Logistic Regression")
    
    # Final Model: Random Forest
    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST (Final Model)")
    print("="*60)
    
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_test_pred_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    
    rf_metrics = print_metrics(y_test, rf_test_pred, rf_test_pred_prob, "Random Forest")
    
    # Model Comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON - TEST SET")
    print("="*60)
    comparison_df = pd.DataFrame({
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics
    })
    print(comparison_df.to_string())
    
    # Save the Random Forest model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_path = Path(__file__).parent.parent / 'models' / 'credit_scoring_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_pipeline, model_path)
    print(f"Random Forest model saved to: {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("To make predictions on a new applicant, run:")
    print("  python src/predict.py")


if __name__ == '__main__':
    main()