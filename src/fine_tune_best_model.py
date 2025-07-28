from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from data_preparation import get_x_transformed_data, get_y_transformed_data
from model_training import random_forest_reg, xgboost_reg

X = get_x_transformed_data()
y = get_y_transformed_data()

rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,
                               cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)

grid_search_rf.fit(X, y)


xgb = XGBRegressor(random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb,
                                cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)

grid_search_xgb.fit(X, y)

print("Best RF params:", grid_search_rf.best_params_)
print("Best RF RMSE:", -grid_search_rf.best_score_)
print("Best XGB params:", grid_search_xgb.best_params_)
print("Best XGB RMSE:", -grid_search_xgb.best_score_)

