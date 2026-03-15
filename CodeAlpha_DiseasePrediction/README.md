# Disease Prediction using Machine Learning

This project builds machine learning models to predict diseases using patient medical data.
It trains classification models on multiple medical datasets and predicts disease risk for new patients.

The models are implemented using **Logistic Regression with an automated preprocessing pipeline from scikit-learn**.

---

## Project Objective

The goal of this project is to:

* Train machine learning models on medical datasets
* Predict disease risk for new patients
* Support multiple datasets with separate trained models
* Provide a simple command line interface for training and prediction

---

## Technologies Used

* Python 3
* pandas
* numpy
* scikit-learn
* joblib

---

## Project Structure

```
CodeAlpha_DiseasePrediction
│
├── data
│   ├── heart.csv
│   ├── diabetes.csv
│   ├── breast_cancer.csv
│   ├── sample_patients_heart.csv
│   ├── sample_patients_diabetes.csv
│   └── sample_patients_breast_cancer.csv
│
├── models
│
├── src
│   ├── train.py
│   ├── predict.py
│   └── data_loader.py
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

It is recommended to create a **virtual environment** before installing dependencies.

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

## Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Required packages:

* pandas
* numpy
* scikit-learn
* joblib

---

## Training Models

Train a machine learning model using a dataset and target column.

```
python src/train.py <dataset_path> <target_column>
```

### Example Commands

Train the heart disease model

```
python src/train.py data/heart.csv target
```

Train the diabetes model

```
python src/train.py data/diabetes.csv target
```

Train the breast cancer model

```
python src/train.py data/breast_cancer.csv target
```

During training the script will:

* Load the dataset
* Handle missing values
* Split the data into training and testing sets
* Apply preprocessing (scaling and encoding)
* Train a Logistic Regression model
* Evaluate the model
* Save the trained model inside the `models` directory

Example output:

```
Accuracy: 0.80
Precision: 0.78
Recall: 0.83
F1 Score: 0.81
```

The trained model will be saved as:

```
models/heart_model.pkl
models/heart_columns.pkl
```

Each dataset creates its **own model**.

---

## Making Predictions

After training a model, you can predict disease risk for new patients.

```
python src/predict.py <dataset_name> [csv_file]
```

### Example Usage

Run predictions using sample patient data (stored as sample_patients.csv inside data):

```
python src/predict.py heart
python src/predict.py diabetes
python src/predict.py breast_cancer
```

Run predictions using a custom dataset:

```
python src/predict.py heart data/my_patients.csv
```

Example prediction output:

```
Disease Prediction Results

Patient 1
Disease Risk: Low
Healthy: 88.81%
Disease: 11.19%

Patient 2
Disease Risk: High
Healthy: 6.92%
Disease: 93.08%
```

---

## Datasets

This project uses three commonly used medical datasets.

### Heart Disease Dataset

Link:
[https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Diabetes Dataset (Pima Indians Diabetes)

Link:
[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Breast Cancer Dataset

Link:
[https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Download the datasets and place them inside the **data folder**.

---

## Key Features

* Train models on multiple datasets
* Separate model for each dataset
* Command line training and prediction
* Automatic preprocessing pipeline
* Predict disease probability for patients
* Works with custom patient datasets

---

## Example Workflow

Train a model

```
python src/train.py data/heart.csv target
```

Run predictions

```
python src/predict.py heart
```

---

## Author

Developed as part of the **CodeAlpha Machine Learning Internship**.

---