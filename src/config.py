import os

# Google Cloud Platform configuration
PROJECT_ID = "ds-club-lulc"
PROJECT_NUMBER = 456110130984
BUCKET_NAME = "dslulc-training-data"

# Data paths
TRAIN_PATH = f'gs://{BUCKET_NAME}/land_use_train.csv'
VAL_PATH = f'gs://{BUCKET_NAME}/land_use_val.csv'

# Earth Engine configuration
EE_DATASET = "ESA/WorldCover/v100"

# Model configuration
BANDS = ['B2', 'B3', 'B4', 'B8']
TARGET_COL = 'lc'
RANDOM_STATE = 42
RF_ESTIMATORS = 500
XGB_ESTIMATORS = 200

# Local paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")

# Image paths (example)
TEST_IMAGE_PATH = '/content/drive/MyDrive/DSTestImages/Test_Image_SanDiego.tif'
MODEL_PKL_PATH = '/content/drive/MyDrive/DSTestImages/landcover_rf_model.pkl'

# ESA WorldCover colors
ESA_WORLDCOVER_COLORS = [
    '#006400', # 0: Forest
    '#ffbb22', # 1: Shrubland
    '#ffff4c', # 2: Grassland
    '#f096ff', # 3: Cropland
    '#fa0000', # 4: Built-up
    '#b4b4b4', # 5: Bare
    '#0064ff'  # 6: Water
]

def get_esa_colormap():
    """
    Create a Matplotlib ListedColormap for ESA WorldCover classes.
    """
    from matplotlib.colors import ListedColormap
    return ListedColormap(ESA_WORLDCOVER_COLORS)
