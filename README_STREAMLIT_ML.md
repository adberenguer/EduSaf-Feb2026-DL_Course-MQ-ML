# Fruit Classification ML Pipeline - Streamlit App

A comprehensive Streamlit web application for training and evaluating machine learning models for fruit classification using MQ sensor data.

## Features

- **Train Models**: Train multiple ML algorithms (ANN/MLP, KNN, Logistic Regression, Random Forest, SVM, optional XGBoost)
- **View Performance**: Visualize model performance with plots and confusion matrices
- **Make Predictions**: Use trained models to predict fruit types from sensor readings

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

1. Run the Streamlit app:
```bash
streamlit run streamlit_ML_app.py
```

2. The app will open in your browser automatically (usually at `http://localhost:8501`)

## Usage

### 1. Train Models Page
- Click "🔄 Train All Models" to train all algorithms
- The app will:
  - Load data from `AllSmaples-Report/` directory
  - Exclude drift files (`D1`–`D5`) automatically
  - Perform a stratified 80/20 train-test split
  - Train each model and evaluate on the held-out test set
  - Save models to `saved_models/` directory

### 2. Model Comparison Page
- View performance comparison charts
- See detailed metrics table
- Visualize confusion matrices for each model
- Identify the best performing model

### 3. Make Prediction Page
- Enter MQ sensor readings (MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135)
- Select a trained model
- Get predictions with confidence scores

## Saved Models

Trained models are automatically saved in the `saved_models/` directory:
- Individual model files: `{model_name}.joblib`
- Metadata file: `metadata.joblib` (contains label mappings, performance metrics, etc.)

## Model Architecture

All models use a preprocessing pipeline with:
- **StandardScaler**: Normalizes features for optimal performance
- **Classifier**: One of the available algorithms (ANN, KNN, LR, RF, SVM, optional XGBoost)

## Data Structure

The app expects CSV files in `AllSmaples-Report/` directory with columns:
- Ticks (timestamp)
- MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135 (sensor readings)
- Label column will be automatically created from filename

## Notes

- Models are trained with a stratified train-test split (80/20)
- XGBoost is optional and only available if `xgboost` is installed
- The best model is automatically identified based on test accuracy
