# Sarcasm Detection using Machine Learning

## Project Overview

This project implements a sarcasm detection system using various machine learning models. The goal is to classify text as sarcastic or non-sarcastic based on a dataset of labeled text samples. The project includes data preprocessing, model training, evaluation, and a frontend interface (in progress) for user interaction. The models evaluated include Gradient Boosting Machine (GBM), CatBoost, XGBoost, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, Light GBM, and AdaBoost.

The project is structured into three main phases:
1. **Model Creation, Training, and Evaluation** [Completed]
2. **Frontend Development with Flask** [In Progress]
3. **Application Deployment** [To Be Done]

## Dataset

The dataset used contains text samples labeled as sarcastic (1) or non-sarcastic (0). It is split into training, validation, and test sets, with indices provided for each subset. The dataset structure includes:
- `info`: Labels for each text sample.
- `texts`: The text data for classification.
- `train_ind`, `val_ind`, `test_ind`: Indices for training, validation, and test splits.

## Model Performance

The following table summarizes the performance of the top 4 models based on accuracy, as evaluated on the test set:

| Model         | Accuracy  | Precision | Recall   | F1 Score |
|---------------|-----------|-----------|----------|----------|
| GBM           | 0.813602  | 0.843391  | 0.813602 | 0.809901 |
| CatBoost      | 0.806045  | 0.821583  | 0.806045 | 0.804016 |
| XGBoost       | 0.788413  | 0.788775  | 0.788413 | 0.788273 |
| Random Forest | 0.770781  | 0.774146  | 0.770781 | 0.770291 |

The performance comparison is visualized in a bar chart (see `01_Data.ipynb` for details).

