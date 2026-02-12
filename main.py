import os
import argparse
from src import config, ee_utils, data_loader, image_processing
from src.models import trained_models as model_utils

def run_pipeline(train=False, classify_image=None):
    """
    Main pipeline for training and prediction.
    """
    # 1. Initialize Earth Engine
    ee_utils.initialize_ee()

    # 2. Train model if requested
    trained_model = None
    if train:
        df_train, df_val = data_loader.load_training_data()
        X_train, y_train = data_loader.get_features_labels(df_train)
        trained_model = model_utils.train_rf(X_train, y_train)
        
        # Save model
        model_save_path = os.path.join(config.MODELS_DIR, "landcover_rf_model.pkl")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model_utils.save_rf(trained_model, model_save_path)
    
    # 3. Classify image if requested
    if classify_image:
        if not trained_model:
            # Attempt to load existing model
            model_path = os.path.join(config.MODELS_DIR, "landcover_rf_model.pkl")
            if os.path.exists(model_path):
                trained_model = model_utils.load_rf(model_path)
            else:
                print("Error: No trained model found and 'train' flag not set.")
                return

        print(f"Classifying image: {classify_image}")
        img, profile = image_processing.load_geotiff(classify_image)
        img_prep, h, w = image_processing.preprocess_image(img)
        
        prediction = image_processing.predict_on_image(trained_model, img_prep)
        prediction_map = prediction.reshape(h, w)
        
        # Save output
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(config.OUTPUT_DIR, "SD_Prediction.tif")
        image_processing.save_prediction_geotiff(prediction_map, profile, output_path)
    else:
        print("No image provided for classification. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LULC Classification Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--classify", type=str, help="Path to GeoTIFF image to classify")
    
    args = parser.parse_args()
    
    run_pipeline(train=args.train, classify_image=args.classify)
