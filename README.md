# LULC Land Use Classification Project

This project refactors the original `ds_datathon_lulc.py` script into a modular, command-line driven Python application for Land Use and Land Cover (LULC) classification.

## Project Structure

```text
LULC/
├── main.py              # Main entry point for the pipeline
├── requirements.txt      # Project dependencies
├── README.md            # This file
├── src/
│   ├── __init__.py      # Package indicator
│   ├── config.py         # Central configuration and constants
│   ├── data_loader.py    # GCS data loading logic
│   ├── ee_utils.py       # Google Earth Engine utilities
│   ├── image_processing.py # GeoTIFF and prediction logic
│   └── models/
│       ├── __init__.py  # Model package init
│       └── trained_models.py # Shared logic for RF and XGBoost models
├── trained_models/       # Directory where trained .pkl files are stored
└── outputs/              # Directory where prediction maps are saved
```

## Setup Instructions

### 1. Activate Virtual Environment
Activate it using:

```bash
source venv/bin/activate
```

### 2. Configure Your Project
Open `src/config.py` to update your Google Cloud Project ID, Bucket Name, or model parameters.

## Usage

You can run the entire pipeline through `main.py`.

### Train the Model
This will load data from GCS, train a Random Forest model, and save it to `trained_models/`.
```bash
python main.py --train
```

### Classify a New Image
This will load a GeoTIFF, use the trained model, and save a prediction map to `outputs/`.
```bash
python main.py --classify /path/to/your/image.tif
```

### Run Both
```bash
python main.py --train --classify /path/to/your/image.tif
```

## Optional: Enable XGBoost
XGBoost is included but optional to avoid dependency issues on macOS. To use it:
1. Install OpenMP: `brew install libomp`
2. Update `main.py` or your training logic to call `model_utils.train_xgb`.
