import os
import argparse
from src import config, ee_utils, data_loader, image_processing
from src.models import trained_models as model_utils

def run_pipeline(train=False, classify_image=None, model_type="rf", model_path=None, visualize=False):
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
        
        if model_type == "rf":
            trained_model = model_utils.train_rf(X_train, y_train)
            model_save_path = model_utils.save_rf(trained_model)
        elif model_type == "xgb":
            trained_model = model_utils.train_xgb(X_train, y_train)
            model_save_path = model_utils.save_xgb(trained_model)
        else:
            print(f"Error: Unsupported model type '{model_type}'")
            return
            
        print(f"Model saved as versioned file: {model_save_path}")
    
    # 3. Classify image if requested
    if classify_image:
        if not trained_model:
            # Determine which model path to use
            actual_model_path = model_path if model_path else model_utils.get_latest_model_path(model_type)
            
            if actual_model_path and os.path.exists(actual_model_path):
                # Infer model type from filename if not explicitly provided or if path is provided
                # However, we rely on the user providing the correct --model flag if using XGB
                # since the loader functions are different.
                if model_type == "rf":
                    trained_model = model_utils.load_rf(actual_model_path)
                elif model_type == "xgb":
                    trained_model = model_utils.load_xgb(actual_model_path)
            else:
                error_msg = f"Error: Model file '{actual_model_path}' not found." if actual_model_path else f"Error: No trained {model_type} model found."
                print(error_msg)
                return

        print(f"Classifying image: {classify_image} using {model_type}")
        img, profile = image_processing.load_geotiff(classify_image)
        img_prep, h, w = image_processing.preprocess_image(img)
        
        prediction = image_processing.predict_on_image(trained_model, img_prep)
        prediction_map = prediction.reshape(h, w)
        
        # Save output
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Save raw GeoTIFF (for GIS)
        output_name = f"SD_Prediction_{model_type}.tif"
        output_path = os.path.join(config.OUTPUT_DIR, output_name)
        image_processing.save_prediction_geotiff(prediction_map, profile, output_path)
        
        # Save visualization (for human viewing)
        if visualize:
            visual_name = f"SD_Prediction_{model_type}_viz.png"
            visual_path = os.path.join(config.OUTPUT_DIR, visual_name)
            image_processing.save_prediction_visual(prediction_map, visual_path)
    else:
        print("No image provided for classification. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LULC Classification Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--classify", type=str, help="Path to GeoTIFF image to classify")
    parser.add_argument("--model", type=str, choices=["rf", "xgb"], default="rf", help="Model type: 'rf' for Random Forest, 'xgb' for XGBoost (default: 'rf')")
    parser.add_argument("--model_path", type=str, help="Path to a specific trained model .pkl file (optional, overrides --model for selection)")
    parser.add_argument("--visualize", action="store_true", help="Save a colorized visualization (PNG) of the prediction")
    
    args = parser.parse_args()
    
    run_pipeline(train=args.train, classify_image=args.classify, model_type=args.model, model_path=args.model_path, visualize=args.visualize)
