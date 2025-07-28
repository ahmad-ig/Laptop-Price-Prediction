import pandas as pd
import numpy as np
from data_preparation import pipeline, log_transform
from fine_tune_best_model import grid_search_rf, grid_search_xgb
from model_training import linear_regression, decision_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle

# Plot Actual vs Predicted
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Price (Log Transformed)")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


laptop_test = pd.read_csv('data//laptop_test_set.csv')
X_test = laptop_test.drop(columns='Price')
y_test = laptop_test['Price'].copy()

X_test_prepared = pipeline().transform(X_test)
y_test_prepared = log_transform(y_test)

rf_model = grid_search_rf.best_estimator_

xgb_model = grid_search_xgb.best_estimator_

lin_reg = linear_regression()

dec_tree = decision_tree()

rf_model_prediction = rf_model.predict(X_test_prepared)

xgb_model_prediction = xgb_model.predict(X_test_prepared)

lin_reg_prediction = lin_reg.predict(X_test_prepared)

dec_tree_prediction = dec_tree.predict(X_test_prepared)

# Random Forest Metric
rf_mse = mean_squared_error(y_test_prepared, rf_model_prediction)
rf_rmse = np.sqrt(rf_mse)
print("\nRF Model Test Set MSE:", rf_mse)
print("\nRF Model Test Set RMSE:", rf_rmse)
plot_actual_vs_predicted(y_test_prepared, rf_model_prediction, "Random Forest: Actual vs Predicted")

# XGB Metric
xgb_mse = mean_squared_error(y_test_prepared, xgb_model_prediction)
xgb_rmse = np.sqrt(xgb_mse)
print("\nXGB Model Test Set MSE:", xgb_mse)
print("\nXGB Model Test Set MSE:", xgb_rmse)
plot_actual_vs_predicted(y_test_prepared, xgb_model_prediction, "XGBoost: Actual vs Predicted")

# Linear Reg Metric
lin_reg_mse = mean_squared_error(y_test_prepared, lin_reg_prediction)
lin_reg_rmse = np.sqrt(rf_mse)
print("\nLIN REG Model Test Set MSE:", lin_reg_mse)
print("\nLIN REG Test Set RMSE:", lin_reg_rmse)
plot_actual_vs_predicted(y_test_prepared, lin_reg_prediction, "Linear Regression: Actual vs Predicted")

# Decision Tree Metric
dec_tree_mse = mean_squared_error(y_test_prepared, dec_tree_prediction)
dec_tree_rmse = np.sqrt(dec_tree_mse)
print("\nTREE Test Set MSE:", dec_tree_mse)
print("\nTREE Test Set RMSE:", dec_tree_rmse)
plot_actual_vs_predicted(y_test_prepared, dec_tree_prediction, "Decision Tree: Actual vs Predicted")

# Save Models Best Models (Random Fores & XGB)
with open("models/rf_model.pkl", "wb") as f:
    cloudpickle.dump(rf_model, f)

with open("models/xgb_model.pkl", "wb") as f:
    cloudpickle.dump(xgb_model, f)

print("\nModels saved as rf_model.pkl and xgb_model.pkl")