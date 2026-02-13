import joblib
import os
import glob
import re
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False
from .. import config

def get_next_version(model_type="rf"):
    """
    Find the next version number for a model type in the models directory.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    pattern = os.path.join(config.MODELS_DIR, f"landcover_{model_type}_model_v*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        return 1
    
    versions = []
    for f in files:
        match = re.search(r'_v(\d+)\.pkl$', f)
        if match:
            versions.append(int(match.group(1)))
    
    return max(versions) + 1 if versions else 1

def get_latest_model_path(model_type="rf"):
    """
    Get the path to the latest version of a model type.
    """
    pattern = os.path.join(config.MODELS_DIR, f"landcover_{model_type}_model_v*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        # Fallback to legacy path if no versioned files exist
        legacy_path = os.path.join(config.MODELS_DIR, f"landcover_{model_type}_model.pkl")
        return legacy_path if os.path.exists(legacy_path) else None
    
    # Sort by version number
    files.sort(key=lambda x: int(re.search(r'_v(\d+)\.pkl$', x).group(1)), reverse=True)
    return files[0]

# Random Forest Logic
def train_rf(X_train, y_train, n_estimators=config.RF_ESTIMATORS, random_state=config.RANDOM_STATE):
    """
    Train a Random Forest classifier.
    """
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_rf(model, filepath=None):
    """
    Save the trained RF model to a file with automatic versioning if no path is provided.
    """
    if filepath is None:
        version = get_next_version("rf")
        filepath = os.path.join(config.MODELS_DIR, f"landcover_rf_model_v{version}.pkl")
    
    print(f"Saving Random Forest model to: {filepath}")
    joblib.dump(model, filepath)
    return filepath

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

def save_xgb(model, filepath=None):
    """
    Save the trained XGBoost model to a file with automatic versioning if no path is provided.
    """
    if filepath is None:
        version = get_next_version("xgb")
        filepath = os.path.join(config.MODELS_DIR, f"landcover_xgb_model_v{version}.pkl")
    
    print(f"Saving XGBoost model to: {filepath}")
    joblib.dump(model, filepath)
    return filepath

def load_xgb(filepath):
    """
    Load a trained XGBoost model from a file.
    """
    print(f"Loading XGBoost model from: {filepath}")
    return joblib.load(filepath)
