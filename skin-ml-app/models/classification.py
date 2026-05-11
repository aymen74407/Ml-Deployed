import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')

# Unified mappings to ensure training and prediction always match
LABEL_MAPS = {
    'Gender':         {'Female': 0, 'Male': 1},
    'Hydration_Level':{'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':      {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':    {'High': 0, 'Low': 1, 'Medium': 2},
}
FEATURE_COLS = ['Age', 'Gender_enc', 'Hydration_Level_enc', 'Oil_Level_enc', 'Sensitivity_enc', 'Humidity', 'Temperature']

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    # Using hardcoded maps for features
    for col, mapping in LABEL_MAPS.items():
        df[col + '_enc'] = df[col].map(mapping)

    # LabelEncoder for the target ONLY
    le = LabelEncoder()
    df['Skin_Type_enc'] = le.fit_transform(df['Skin_Type'])

    X = df[FEATURE_COLS].values
    y = df['Skin_Type_enc'].values
    return X, y, le

def encode_input(inp):
    # This must strictly match FEATURE_COLS order
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
        return KNeighborsClassifier(n_neighbors=int(p.get('n_neighbors', 17)), 
                                    weights=p.get('weights', 'distance'),
                                    metric=p.get('metric', 'euclidean')), True
    elif algorithm == 'decision_tree':
        return DecisionTreeClassifier(max_depth=int(p.get('max_depth', 7)), 
                                      ccp_alpha=float(p.get('ccp_alpha', 0.000944)),
                                      min_samples_split=int(p.get('min_samples_split', 10)),
                                      min_samples_leaf=int(p.get('min_samples_leaf', 5)),
                                      random_state=42), False
    elif algorithm == 'random_forest':
        return RandomForestClassifier(n_estimators=int(p.get('n_estimators', 100)), 
                                      max_features=p.get('max_features', 'sqrt'),
                                      random_state=42), False
    elif algorithm == 'svm':
        return SVC(kernel=p.get('kernel', 'rbf'), C=float(p.get('C', 1.0)),
                   gamma=float(p.get('gamma', 0.1)),
                   probability=True, random_state=42), True
    elif algorithm == 'xgboost':
        return xgb.XGBClassifier(n_estimators=int(p.get('n_estimators', 100)),
                                 max_depth=int(p.get('max_depth', 7)),
                                 learning_rate=float(p.get('learning_rate', 0.3)),
                                 subsample=float(p.get('subsample', 0.8)),
                                 colsample_bytree=float(p.get('colsample_bytree', 0.5)),
                                 gamma=float(p.get('gamma', 1.0)),
                                 random_state=42, eval_metric='mlogloss'), False
    raise ValueError(f'Unknown algorithm: {algorithm}')

def train_classifier(algorithm, params=None):
    X, y, le = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

    model, needs_scale = _get_model(algorithm, params)
    model.fit(X_train_s if needs_scale else X_train, y_train)
    y_pred = model.predict(X_test_s if needs_scale else X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    # Feature Importance for Random Forest / XGBoost
    fi = None
    if hasattr(model, 'feature_importances_'):
        fi = dict(zip(['Age', 'Gender', 'Hydration', 'Oil', 'Sensitivity', 'Humidity', 'Temperature'],
                       [round(float(v)*100, 2) for v in model.feature_importances_]))

    return {
        'accuracy': round(acc * 100, 2),
        'feature_importance': fi,
        'classes': le.classes_.tolist(),
    }

# Cache for models to improve performance
_MODEL_CACHE = {}

def predict_single(algorithm, encoded_features, params=None):
    cache_key = f"{algorithm}_{str(params)}"
    X, y, le = load_and_preprocess()
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
    skin_type = le.inverse_transform([pred])[0]
    
    proba_dict = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(x_in)[0]
        for cls, p in zip(le.classes_, proba): 
            proba_dict[cls] = round(float(p)*100, 1)
            
    return {'skin_type': skin_type, 'probabilities': proba_dict}
