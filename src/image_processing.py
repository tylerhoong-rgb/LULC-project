import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from . import config

def load_geotiff(file_path):
    """
    Load a GeoTIFF image and return its data and profile.
    """
    with rasterio.open(file_path) as src:
        img = src.read()
        profile = src.profile
    return img, profile

def preprocess_image(img):
    """
    Prepare GeoTIFF image for prediction.
    Returns a pandas DataFrame to maintain feature names.
    """
    if img.shape[0] != 4:
         print(f"Warning: Expected 4 bands, but got {img.shape[0]} bands.")
    
    img = img.astype(np.float32)
    h, w = img.shape[1], img.shape[2]
    img_reshaped = img.transpose(1, 2, 0).reshape(-1, 4)
    img_reshaped = np.nan_to_num(img_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to DataFrame with consistent band names
    df = pd.DataFrame(img_reshaped, columns=config.BANDS)
    return df, h, w

def calculate_ndvi(img):
    """
    Calculate NDVI from B4 (red) and B8 (NIR).
    Assumes bands are indexed as [B2, B3, B4, B8].
    """
    red = img[2]
    nir = img[3]
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

def predict_on_image(model, img_data):
    """
    Run prediction on preprocessed image data.
    """
    prediction = model.predict(img_data)
    return prediction

def save_prediction_geotiff(prediction_map, profile, output_path):
    """
    Save the prediction map as a new GeoTIFF.
    """
    new_profile = profile.copy()
    new_profile.update(dtype=rasterio.int32, count=1, compress='lzw')
    
    with rasterio.open(output_path, 'w', **new_profile) as dst:
        dst.write(prediction_map.astype(np.int32), 1)
    print(f"Prediction saved to: {output_path}")

def visualize_prediction(prediction_map, title="Land Use Classification"):
    """
    Visualize the prediction map.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(prediction_map, cmap='terrain')
    plt.colorbar(label='Land Use Class')
    plt.title(title)
    plt.show()
