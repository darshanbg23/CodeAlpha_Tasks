import pandas as pd
import joblib
import sys
import os


def validate_model_files(dataset_name):
    model_file = f'models/{dataset_name}_model.pkl'
    columns_file = f'models/{dataset_name}_columns.pkl'
    
    if not os.path.exists(model_file):
        print(f"Error: Model for dataset '{dataset_name}' not found at {model_file}")
        print(f"\nPlease train it first using:")
        print(f"  python src/train.py data/{dataset_name}.csv <target_column>")
        sys.exit(1)
    
    if not os.path.exists(columns_file):
        print(f"Error: Training columns metadata not found at {columns_file}")
        print(f"Please train the model first.")
        sys.exit(1)
    
    return model_file, columns_file


def validate_dataset(csv_file, training_columns):
    if not os.path.exists(csv_file):
        print(f"Error: Prediction dataset not found at {csv_file}")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    missing_cols = set(training_columns) - set(df.columns)
    if missing_cols:
        print(f"Error: Missing required columns in prediction dataset:")
        print(f"  Missing: {sorted(missing_cols)}")
        print(f"  Required: {sorted(training_columns)}")
        print(f"  Found: {sorted(df.columns)}")
        sys.exit(1)
    
    return df[training_columns]


def predict_from_data(model, data_df):
    predictions = model.predict(data_df)
    probabilities = model.predict_proba(data_df)
    return predictions, probabilities


def format_probability(prob):
    return f"{prob * 100:.2f}%"


def main(dataset_name, csv_file=None):
    model_file, columns_file = validate_model_files(dataset_name)
    
    model = joblib.load(model_file)
    training_columns = joblib.load(columns_file)
    
    df = validate_dataset(csv_file, training_columns)
    predictions, probabilities = predict_from_data(model, df)
    
    print("\n" + "="*60)
    print(f"Disease Prediction Results - {dataset_name.upper()}")
    print("="*60)
    
    for i, (pred, probs) in enumerate(zip(predictions, probabilities), 1):
        risk_level = 'High' if pred == 1 else 'Low'
        healthy_prob = format_probability(probs[0])
        disease_prob = format_probability(probs[1])
        
        print(f"\nPatient {i} - Disease Risk: {risk_level}")
        print(f"Healthy: {healthy_prob}")
        print(f"Disease: {disease_prob}")
    
    print("\n" + "="*60 + "\n")


def print_usage():
    print("\nUsage: python predict.py <dataset_name> [csv_file]")
    print("\nExample:")
    print("  python predict.py heart")
    print("  python predict.py heart data/sample_patients_heart.csv")
    print("  python predict.py diabetes data/sample_patients_diabetes.csv")
    print("  python predict.py breast_cancer data/sample_patients_cancer.csv")
    print("\nAvailable datasets: heart, diabetes, breast_cancer")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_usage()
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) == 3 else f'data/sample_patients_{dataset_name}.csv'
    
    main(dataset_name, csv_file)