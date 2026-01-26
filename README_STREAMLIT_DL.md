# Fruit Quality Monitoring - Deep Learning Dashboard

A Streamlit web application for visualizing and comparing pre-trained deep learning model configurations for fruit quality monitoring using MQ sensor data.

## Features

- **📊 Model Comparison**: Compare multiple pre-trained model configurations side-by-side
- **🔍 Training Analysis**: View training curves, confusion matrices, and classification reports for each configuration
- **🔮 Make Predictions**: Use any pre-trained configuration to make predictions on new sensor data
- **📦 Configuration Management**: Load and compare different hyperparameter configurations

## Important Note

**This dashboard does NOT train models.** All models are pre-trained in the Jupyter notebook (`course_example_predicting_Non_dest_fruit_quality_monitoring.ipynb`) and saved to `saved_models_pytorch/`. This app only loads, visualizes, and uses pre-trained models.

## Installation

1. Install dependencies:
```bash
pip install streamlit torch plotly pandas numpy scikit-learn
```

Or use the requirements file (if available):
```bash
pip install -r requirements.txt
```

## Running the App

1. **First, train models in the notebook:**
   - Open `course_example_predicting_Non_dest_fruit_quality_monitoring.ipynb`
   - Run all cells to train multiple model configurations
   - Models will be saved to `saved_models_pytorch/` directory

2. **Then run the Streamlit app:**
```bash
streamlit run streamlit_DL_app.py
```

3. The app will open in your browser automatically (usually at `http://localhost:8501`)

## Usage

### 🏠 Home Page
- Overview of loaded configurations
- List of all available model configurations
- Information about each configuration

### 📊 Model Comparison Page
- Side-by-side comparison of all configurations
- Interactive bar charts comparing metrics (Accuracy, Precision, Recall, F1-Score)
- Detailed metrics table
- Compare test set performance across all configurations

### 🔍 Training Analysis Page
- Select a specific configuration to analyze
- View training curves (loss and accuracy over epochs)
- Interactive confusion matrix for test set
- Classification report with per-class metrics

### 🔮 Make Prediction Page
- Upload CSV file with sensor data
- Select a configuration to use for prediction
- View predicted freshness class
- Interactive probability visualization
- Proper preprocessing is automatically applied using saved configurations

## File Structure

```
.
├── streamlit_DL_app.py              # Streamlit dashboard application
├── course_example_predicting_Non_dest_fruit_quality_monitoring.ipynb  # Training notebook
├── saved_models_pytorch/            # Saved model configurations
│   ├── cnn1d_base.pth
│   ├── cnn1d_deep.pth
│   ├── cnn1d_wide.pth
│   ├── cnn_lstm_base.pth
│   ├── cnn_lstm_large.pth
│   ├── cnn_transformer_base.pth
│   ├── cnn_transformer_large.pth
│   └── preprocessing_scaler.pkl     # Saved scaler (if normalization used)
├── training_results.pkl             # Training results for all configurations
└── README_STREAMLIT_DL.md          # This file
```

## Model Configurations

Each saved model includes:
- **Model architecture** (CNN1D, CNN_LSTM, or CNN_Transformer)
- **Hyperparameters** (filters, kernels, activation functions, dropout rates, etc.)
- **Preprocessing configuration** (normalization method, drift handling, windowing)
- **Scaler object** (if normalization was used during training)
- **Configuration name** for easy identification

## Configuration Naming

Configurations are saved with descriptive names:
- `cnn1d_base`: Base CNN1D configuration
- `cnn1d_deep`: Deeper CNN1D with more layers
- `cnn1d_wide`: Wider CNN1D with more filters
- `cnn_lstm_base`: Base CNN+LSTM configuration
- `cnn_lstm_large`: Larger CNN+LSTM with more layers
- `cnn_transformer_base`: Base CNN+Transformer configuration
- `cnn_transformer_large`: Larger CNN+Transformer with more attention heads

## Data Structure

For predictions, the app expects CSV files with:
- Sensor columns (MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135)
- Each row represents a time step
- The app automatically handles preprocessing based on the saved configuration

## Tips

1. **Training**: Train all configurations in the notebook first before using the dashboard
2. **Comparison**: Use the Model Comparison page to identify the best performing configuration
3. **Analysis**: Use Training Analysis to understand model behavior and identify potential issues
4. **Predictions**: The app automatically applies the correct preprocessing for each configuration

## Troubleshooting

- **No models found**: Make sure you've trained and saved models in the notebook first
- **Missing scaler**: If you used normalization, ensure `preprocessing_scaler.pkl` exists
- **Prediction errors**: Check that your input data matches the expected format (sensor columns)

## Notes

- Models are loaded with caching for better performance
- All preprocessing is handled automatically based on saved configurations
- The dashboard is read-only - no training or model modification happens here
- Training results are loaded from `training_results.pkl` if available
