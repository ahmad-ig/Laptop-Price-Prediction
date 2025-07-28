# Laptop Price Prediction

This project builds a machine learning pipeline to predict laptop prices based on their specifications. It involves data preprocessing, feature engineering, model training, evaluation, and deployment using Flask.

## Project Overview

The goal is to accurately predict the "price of laptops" using features such as:

- Brand
- Processor type
- RAM
- Storage (HDD/SSD/Hybrid)
- Screen resolution
- GPU
- Weight
- Operating System
- Touchscreen

The pipeline supports both training and deployment, with support for multiple regression models.


## Features

- Data preprocessing & cleaning
- Feature engineering with custom transformers
- One-hot encoding of categorical features
- Support for multiple models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Gradient Descent (SGD)
- Model evaluation using RMSE, MSE, Cross-validation
- Hyperparameter tuning (GridSearchCV)
- Deployment using Flask web API
- Scalable pipeline with "scikit-learn" Pipelines and "ColumnTransformers"

## Project Structure

LaptopPricePrediction/
│
├── data/
│ └── laptop_train_set.csv
│ └── laptop_test_set.csv
│ └── laptop_data.csv
│
├── models/
│ └── rf_model.pkl
│ └── xgb_model.pkl
│ └── laptop_preprocessor.pkl
│
├── src/
│ ├── train_test_split.py
│ ├── data_preparation.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── fine_tune_best_model.py
│ ├── model_testing.py
│
├── notebooks/
│ ├── filtering laptop_data.ipynb laptop_report.html
│ ├── laptop_report.html (using ydata_profiling)
│
├── app/
│ ├── app.py
│ ├── templates/
│ 	└── index.html
│ 	└── result.html



| Model            | RMSE       | MSE        |
| ---------------- | ---------- | ---------- |
| LinearRegression | 0.2885     | 0.0832     |      # 9.82% off (5885.7 error)
| Decision Tree    | 0.2857     | 0.0816     |      # 9.73% off (5838.3 error)
| Random Forest    | 0.2334     | 0.0545     |      # 7.92% off (4751.0 error)
| XGBoost          | 0.2191     | 0.0480     |      # 7.46% off (4475.6 error)



## API Usage
Once the Flask app is running, you can send a POST request to /predict with laptop features to get the predicted price.




