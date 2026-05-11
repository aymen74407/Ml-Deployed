import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')

# Unified mappings (Must match classification.py)
LABEL_MAPS = {
    'Gender':         {'Female': 0, 'Male': 1},
    'Hydration_Level':{'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':      {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':    {'High': 0, 'Low': 1, 'Medium': 2},
}
FEATURE_COLS = ['Age', 'Gender_enc', 'Hydration_Level_enc', 'Oil_Level_enc', 'Sensitivity_enc', 'Humidity', 'Temperature']

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    for col, mapping in LABEL_MAPS.items():
        df[col + '_enc'] = df[col].map(mapping)
    return df

def encode_input(inp):
    return [
        float(inp.get('Age', 30)),
        LABEL_MAPS['Gender'].get(inp.get('Gender', 'Female'), 0),
        LABEL_MAPS['Hydration_Level'].get(inp.get('Hydration_Level', 'Medium'), 2),
        LABEL_MAPS['Oil_Level'].get(inp.get('Oil_Level', 'Medium'), 2),
        LABEL_MAPS['Sensitivity'].get(inp.get('Sensitivity', 'Low'), 1),
        float(inp.get('Humidity', 50.0)),
        float(inp.get('Temperature', 25.0)),
    ]

def _get_model(algorithm, params):
    p = params or {}
    if algorithm == 'knn':
        return KNeighborsRegressor(n_neighbors=int(p.get('n_neighbors', 17)), 
                                   weights=p.get('weights', 'distance')), True
    elif algorithm == 'decision_tree':
        return DecisionTreeRegressor(max_depth=int(p.get('max_depth', 7)), 
                                     ccp_alpha=float(p.get('ccp_alpha', 0.000944)),
                                     random_state=42), False
    elif algorithm == 'random_forest':
        return RandomForestRegressor(n_estimators=int(p.get('n_estimators', 100)), 
                                     max_features=p.get('max_features', 'sqrt'),
                                     random_state=42), False
    elif algorithm == 'svm':
        return SVR(kernel=p.get('kernel', 'rbf'), C=float(p.get('C', 1.0)), 
                   gamma=float(p.get('gamma', 0.1))), True
    elif algorithm == 'xgboost':
        return xgb.XGBRegressor(n_estimators=int(p.get('n_estimators', 100)),
                                max_depth=int(p.get('max_depth', 7)),
                                learning_rate=float(p.get('learning_rate', 0.3)),
                                random_state=42), False
    raise ValueError(f'Unknown algorithm: {algorithm}')

_MODEL_CACHE = {}

def predict_single_regression(algorithm, target_col, encoded_features, params=None):
    cache_key = f"{algorithm}_{target_col}_{str(params)}"
    df = load_and_preprocess()
    X = df[FEATURE_COLS].values
    y = df[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if cache_key in _MODEL_CACHE:
        model, needs_scale = _MODEL_CACHE[cache_key]
    else:
        model, needs_scale = _get_model(algorithm, params)
        model.fit(X_scaled if needs_scale else X, y)
        _MODEL_CACHE[cache_key] = (model, needs_scale)
    
    x_in = scaler.transform([encoded_features]) if needs_scale else [encoded_features]
    pred = model.predict(x_in)[0]
    
    return {'prediction': round(float(pred), 2), 'target': target_col}

def train_regressor(algorithm, target_col, params=None):
    df = load_and_preprocess()
    X = df[FEATURE_COLS].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

    model, needs_scale = _get_model(algorithm, params)
    model.fit(X_train_s if needs_scale else X_train, y_train)
    y_pred = model.predict(X_test_s if needs_scale else X_test)

    return {
        'r2': round(r2_score(y_test, y_pred), 4),
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'target': target_col
    }
