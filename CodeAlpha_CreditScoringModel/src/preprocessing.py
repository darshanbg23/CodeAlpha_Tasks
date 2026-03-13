from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_features(df):
    categorical_features = [
        'checking_account',
        'credit_history',
        'purpose',
        'savings_account',
        'employment_since',
        'personal_status',
        'other_debtors',
        'property',
        'other_installments',
        'housing',
        'job',
        'telephone',
        'foreign_worker'
    ]
    
    numerical_features = [
        'duration_months',
        'credit_amount',
        'installment_rate',
        'residence_since',
        'age',
        'existing_credits',
        'dependents'
    ]
    
    return categorical_features, numerical_features


def build_preprocessor(categorical_features, numerical_features):
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ]
    )
    
    return preprocessor