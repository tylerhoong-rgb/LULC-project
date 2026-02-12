import pandas as pd
from . import config

def load_training_data():
    """
    Load training and validation data from GCS.
    """
    print(f"Loading training data from: {config.TRAIN_PATH}")
    df_train = pd.read_csv(config.TRAIN_PATH)
    print(f"Loading validation data from: {config.VAL_PATH}")
    df_val = pd.read_csv(config.VAL_PATH)
    return df_train, df_val

def get_features_labels(df, bands=config.BANDS, target=config.TARGET_COL):
    """
    Separate features and labels.
    """
    X = df[bands]
    y = df[target]
    return X, y
