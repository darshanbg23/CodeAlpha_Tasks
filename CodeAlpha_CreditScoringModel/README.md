# Credit Scoring Model using Machine Learning

This project builds a machine learning model to predict whether a loan applicant is likely to have **good credit or bad credit** based on their financial history.

The model is trained using the **German Credit Dataset** and uses **Logistic Regression and Random Forest classifiers implemented with a preprocessing pipeline from scikit-learn**.

---

# Project Objective

The goal of this project is to:

* Train machine learning models to evaluate creditworthiness
* Predict whether a loan applicant has good or bad credit
* Use financial and personal history features for prediction
* Provide a simple command line interface for training and prediction

---

# Technologies Used

* Python 3
* pandas
* numpy
* scikit-learn
* joblib

---

# Project Structure

```
CodeAlpha_CreditScoringModel
│
├── data
│   ├── german.data
│   └── sample_applicants.csv
│
├── models
│   └── credit_scoring_model.pkl
│
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

# Environment Setup

It is recommended to create a **virtual environment** before installing dependencies.

### Create Virtual Environment

```
python -m venv venv
```

### Activate Environment

**Windows**

```
venv\Scripts\activate
```

**Mac / Linux**

```
source venv/bin/activate
```

---

# Install Dependencies

Install the required libraries using:

```
pip install -r requirements.txt
```

Required packages:

* pandas
* numpy
* scikit-learn
* joblib

---

# Training the Model

To train the credit scoring model run:

```
python src/train.py
```

During training the script will:

* Load the German Credit dataset
* Split the data into training and testing sets
* Identify categorical and numerical features
* Apply preprocessing (OneHotEncoding and scaling)
* Train Logistic Regression and Random Forest models
* Evaluate both models using classification metrics
* Save the final Random Forest model

Example output:

```
Accuracy: 0.82
Precision: 0.84
Recall: 0.79
F1 Score: 0.81
ROC-AUC: 0.87
```

The trained model will be saved as:

```
models/credit_scoring_model.pkl
```

---

# Making Predictions

After training the model, predictions can be made for new applicants using a CSV file.

```
python src/predict.py data/sample_applicants.csv
```

The script will:

* Load the trained model
* Read applicant data from the CSV file
* Predict creditworthiness
* Display probability scores

Example prediction output:

```
Applicant 1
Prediction: Good Credit
Probability (Good): 85.23%
Probability (Bad): 14.77%

Applicant 2
Prediction: Bad Credit
Probability (Good): 32.11%
Probability (Bad): 67.89%
```

---

# Dataset

This project uses the **German Credit Dataset**, which contains information about loan applicants.

Link:

[https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29)

Dataset characteristics:

* 1000 loan applicants
* 20 input features
* Features include financial history, employment status, savings accounts, and personal information
* Target variable indicates **good credit or bad credit**

Download the dataset and place it inside the **data folder**.

---

# Key Features

* Creditworthiness prediction using machine learning
* Feature engineering with preprocessing pipelines
* Two model comparison (Logistic Regression and Random Forest)
* Command line training and prediction
* Probability-based predictions
* Batch prediction using CSV files

---

# Example Workflow

Train the model

```
python src/train.py
```

Predict applicant creditworthiness

```
python src/predict.py data/sample_applicants.csv
```

---

# Author

Developed as part of the **CodeAlpha Machine Learning Internship**.

---