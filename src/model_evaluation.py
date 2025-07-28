from data_preparation import get_x_transformed_data, get_y_transformed_data
from model_training import linear_regression, gradient_descent_regression, decision_tree, random_forest_reg, xgboost_reg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Linear Regression Metric Function
def lin_reg_metric(model, X, y):
    y_predict = model.predict(X)

    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    lin_scores = cross_val_score(model, X, y,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("\nCross Validation scores:")
    display_scores(lin_rmse_scores)

    return y, y_predict

# Gradient Descent Regression Metric Function
def gradint_descent_reg(model, X_scaled, y):
    y_predict = model.predict(X_scaled)

    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    return y, y_predict

# Decision tree Metric Function
def tree_metric(model, X, y):
    y_predict = model.predict(X)

    tree_mse = mean_squared_error(y, y_predict)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Mean Squared Error: {tree_mse}")
    print(f"Root Mean Squared Error: {tree_rmse}")

    tree_scores = cross_val_score(model, X, y,
                                  scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    display_scores(tree_rmse_scores)

    return y, y_predict

# Random Fores Metric Function
def forest_metric(model, X, y):
    y_predict = model.predict(X)

    forest_mse = mean_squared_error(y, y_predict)
    forest_rmse = np.sqrt(forest_mse)
    print(f"Mean Squared Error: {forest_mse}")
    print(f"Root Mean Squared Error: {forest_rmse}")

    forest_scores = cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    return y, y_predict

# XGB Metric Function
def xgboost_metric(model, X, y):
    y_predict = model.predict(X)

    xgb_mse = mean_squared_error(y, y_predict)
    xgb_rmse = np.sqrt(xgb_mse)
    print(f"Mean Squared Error: {xgb_mse}")
    print(f"Root Mean Squared Error: {xgb_rmse}")

    xgb_scores = cross_val_score(model, X, y,
                                 scoring="neg_mean_squared_error", cv=10)
    xgb_rmse_scores = np.sqrt(-xgb_scores)
    print("\nCross Validation scores:")
    display_scores(xgb_rmse_scores)
    
    return y, y_predict


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


X = get_x_transformed_data()
y = get_y_transformed_data()

# Linear Regression
print("\nLINEAR REGRESSION EVALUATION")
trained_lin_reg = linear_regression()
y_true, y_pred = lin_reg_metric(trained_lin_reg, X, y)
plot_actual_vs_predicted(y_true, y_pred, "Linear Regression: Actual vs Predicted")

# Gradient Descent
print("\nGRADIENT DESCENT EVALUATION")
trained_sgd_reg, sgd_scaler = gradient_descent_regression()
X_train_scaled = sgd_scaler.transform(X)
y_true, y_pred = gradint_descent_reg(trained_sgd_reg, X_train_scaled, y)
plot_actual_vs_predicted(y_true, y_pred, "Gradient Descent: Actual vs Predicted")

# Decision Tree
print("\nDECISION TREE EVALUATION")
trained_tree_reg = decision_tree()
y_true, y_pred = tree_metric(trained_tree_reg, X, y)
plot_actual_vs_predicted(y_true, y_pred, "Decision Tree: Actual vs Predicted")

# Random Forest
print("\nRANDOM FOREST EVALUATION")
trained_forest_reg = random_forest_reg()
y_true, y_pred = forest_metric(trained_forest_reg, X, y)
plot_actual_vs_predicted(y_true, y_pred, "Random Forest: Actual vs Predicted")

# XGBoost
print("\nXGBOOST EVALUATION")
trained_xgb_reg = xgboost_reg()
y_true, y_pred = xgboost_metric(trained_xgb_reg, X, y)
plot_actual_vs_predicted(y_true, y_pred, "XGBoost: Actual vs Predicted")
