import pandas as pd
import numpy as np
from data_preparation import get_x_transformed_data, get_y_transformed_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



def load_data():
    X_train = get_x_transformed_data()
    y_train = get_y_transformed_data()
    return X_train, y_train

def linear_regression():
    X_train, y_train = load_data()
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg

def gradient_descent_regression():
    X_train, y_train = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01, random_state=42)
    sgd_reg.fit(X_scaled, y_train)
    return sgd_reg, scaler

def decision_tree():
    X_train, y_train = load_data()
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    return tree_reg

def random_forest_reg():
    X_train, y_train = load_data()
    forest_reg = RandomForestRegressor()
    forest_reg.fit(X_train, y_train)
    return forest_reg

def xgboost_reg():
    X_train, y_train = load_data()
    xg_boost_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xg_boost_reg.fit(X_train, y_train)
    return xg_boost_reg