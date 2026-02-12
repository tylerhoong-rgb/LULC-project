import joblib
import os
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False
from .. import config

# Random Forest Logic
def train_rf(X_train, y_train, n_estimators=config.RF_ESTIMATORS, random_state=config.RANDOM_STATE):
    """
    Train a Random Forest classifier.
    """
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_rf(model, filepath):
    """
    Save the trained RF model to a file.
    """
    print(f"Saving Random Forest model to: {filepath}")
    joblib.dump(model, filepath)

def load_rf(filepath):
    """
    Load a trained RF model from a file.
    """
    print(f"Loading Random Forest model from: {filepath}")
    return joblib.load(filepath)

# XGBoost Logic
def train_xgb(X_train, y_train, n_estimators=config.XGB_ESTIMATORS):
    """
    Train an XGBoost classifier.
    """
    if not HAS_XGB:
        raise ImportError("XGBoost is not properly installed or libomp is missing. Run 'brew install libomp' on Mac.")
    print(f"Training XGBoost with {n_estimators} estimators...")
    model = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def save_xgb(model, filepath):
    """
    Save the trained XGBoost model to a file.
    """
    print(f"Saving XGBoost model to: {filepath}")
    joblib.dump(model, filepath)

def load_xgb(filepath):
    """
    Load a trained XGBoost model from a file.
    """
    print(f"Loading XGBoost model from: {filepath}")
    return joblib.load(filepath)
