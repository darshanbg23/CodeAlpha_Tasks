import sys
import joblib
import pandas as pd
from pathlib import Path

def load_model():
    model_path = Path(__file__).parent.parent / 'models' / 'credit_scoring_model.pkl'
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'python src/train.py' first to train and save the model.")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_creditworthiness(applicant_data, model):
    # Create DataFrame if input is a dictionary
    if isinstance(applicant_data, dict):
        df_applicant = pd.DataFrame([applicant_data])
    else:
        df_applicant = pd.DataFrame([applicant_data])
    
    try:
        prediction = model.predict(df_applicant)[0]
        probabilities = model.predict_proba(df_applicant)[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Make sure all required features are provided.")
        return None
    
    return {
        'prediction': prediction,
        'prob_bad': probabilities[0],
        'prob_good': probabilities[1]
    }


def print_prediction_result(result, applicant_index=None):
    if result is None:
        return
    
    prediction_label = "Good Credit" if result['prediction'] == 1 else "Bad Credit"
    
    if applicant_index is not None:
        print(f"\nApplicant {applicant_index}")
    else:
        print("\n" + "="*50)
        print("CREDIT PREDICTION RESULT")
        print("="*50)
    
    print(f"Prediction: {prediction_label}")
    print(f"Probability (Good): {result['prob_good']:.2%}")
    print(f"Probability (Bad):  {result['prob_bad']:.2%}")
    
    if applicant_index is None:
        print("="*50 + "\n")


def predict_from_csv(csv_path, model):
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if dataframe is empty
    if df.empty:
        print("Error: CSV file is empty.")
        return
    
    print(f"\nLoading model and making predictions for {len(df)} applicant(s)...")
    print("="*50)
    
    # Make predictions for each row
    for idx, row in df.iterrows():
        applicant_num = idx + 1
        result = predict_creditworthiness(row, model)
        if result:
            print_prediction_result(result, applicant_index=applicant_num)
        else:
            print(f"\nApplicant {applicant_num}: Prediction failed")
    
    print("="*50 + "\n")


def print_usage():
    print("Usage: python src/predict.py <csv_file>")
    print()
    print("Example:")
    print("  python src/predict.py data/sample_applicants.csv")
    print()
    print("You must provide a CSV file containing applicant data.")
    print()
    print("Example CSV structure:")
    print()
    print("checking_account,duration_months,credit_history,purpose,credit_amount,savings_account,employment_since,installment_rate,personal_status,other_debtors,residence_since,property,age,other_installments,housing,existing_credits,job,dependents,telephone,foreign_worker")
    print("A12,24,A32,A43,3000,A61,A73,4,A93,A101,2,A121,35,A143,A152,1,A173,1,A192,A201")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    csv_path = sys.argv[1]
    model = load_model()
    if model is None:
        return
    
    predict_from_csv(csv_path, model)


if __name__ == '__main__':
    main()

# Note: If you see "Found unknown categories" warning, it means your CSV
# contains category values not seen in training data. The model handles this
# gracefully. To suppress this warning (optional), add to the top of the file:
#   import warnings
#   warnings.filterwarnings('ignore', message='Found unknown categories')