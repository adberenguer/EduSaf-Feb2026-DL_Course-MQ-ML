import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import warnings
import io
import re
warnings.filterwarnings('ignore')

# Set page config with fancy theme
st.set_page_config(
    page_title="🍎 Fruit Quality Monitoring - Deep Learning Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy UI
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00c851;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff8800;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for activation functions
def get_activation(activation_name='relu'):
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'tanh': nn.Tanh(),
        'swish': nn.SiLU(),  # SiLU is Swish
        'mish': nn.Mish() if hasattr(nn, 'Mish') else nn.ReLU()
    }
    return activations.get(activation_name.lower(), nn.ReLU())

# Model architectures (same as in notebook with hyperparameters)
class CNN1D(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length,
                 filters=[64, 128, 256], kernel_sizes=[5, 5, 3],
                 fc_sizes=[128, 64], dropout_conv=0.3, dropout_fc=0.5,
                 activation='relu', use_batch_norm=True, pool_size=2):
        super(CNN1D, self).__init__()
        self.activation_fn = get_activation(activation)
        self.use_batch_norm = use_batch_norm
        
        num_conv_layers = max(len(filters), len(kernel_sizes))
        if len(filters) < num_conv_layers:
            filters = filters + [filters[-1]] * (num_conv_layers - len(filters))
        if len(kernel_sizes) < num_conv_layers:
            kernel_sizes = kernel_sizes + [kernel_sizes[-1]] * (num_conv_layers - len(kernel_sizes))
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = input_channels
        current_length = sequence_length
        
        for i, (out_channels, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            padding = kernel_size // 2
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            self.dropout_layers.append(nn.Dropout(dropout_conv))
            in_channels = out_channels
            current_length = current_length // pool_size
        
        self.flattened_size = filters[-1] * current_length
        
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        fc_input_size = self.flattened_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            self.fc_dropouts.append(nn.Dropout(dropout_fc))
            fc_input_size = fc_size
        
        self.fc_out = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation_fn(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        x = x.view(x.size(0), -1)
        for fc_layer, dropout in zip(self.fc_layers, self.fc_dropouts):
            x = self.activation_fn(fc_layer(x))
            x = dropout(x)
        x = self.fc_out(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length,
                 cnn_filters=[64, 128], cnn_kernel_sizes=[5, 5],
                 lstm_hidden=128, lstm_layers=2, lstm_bidirectional=True,
                 fc_sizes=[128, 64], dropout_conv=0.3, dropout_lstm=0.3, dropout_fc=0.5,
                 activation='relu', use_batch_norm=True, pool_size=2):
        super(CNN_LSTM, self).__init__()
        self.activation_fn = get_activation(activation)
        self.use_batch_norm = use_batch_norm
        
        num_cnn_layers = max(len(cnn_filters), len(cnn_kernel_sizes))
        if len(cnn_filters) < num_cnn_layers:
            cnn_filters = cnn_filters + [cnn_filters[-1]] * (num_cnn_layers - len(cnn_filters))
        if len(cnn_kernel_sizes) < num_cnn_layers:
            cnn_kernel_sizes = cnn_kernel_sizes + [cnn_kernel_sizes[-1]] * (num_cnn_layers - len(cnn_kernel_sizes))
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = input_channels
        current_length = sequence_length
        
        for out_channels, kernel_size in zip(cnn_filters, cnn_kernel_sizes):
            padding = kernel_size // 2
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            self.dropout_layers.append(nn.Dropout(dropout_conv))
            in_channels = out_channels
            current_length = current_length // pool_size
        
        self.cnn_output_length = current_length
        self.cnn_output_channels = cnn_filters[-1]
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )
        
        lstm_output_size = (2 if lstm_bidirectional else 1) * lstm_hidden
        
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        fc_input_size = lstm_output_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            self.fc_dropouts.append(nn.Dropout(dropout_fc))
            fc_input_size = fc_size
        
        self.fc_out = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation_fn(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        x = x.transpose(1, 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        for fc_layer, dropout in zip(self.fc_layers, self.fc_dropouts):
            x = self.activation_fn(fc_layer(x))
            x = dropout(x)
        x = self.fc_out(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNN_Transformer(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length,
                 cnn_filters=[64, 128], cnn_kernel_sizes=[5, 5],
                 d_model=128, nhead=8, num_layers=2, dim_feedforward=512,
                 fc_sizes=[128, 64], dropout_conv=0.3, dropout_transformer=0.1, dropout_fc=0.5,
                 activation='relu', use_batch_norm=True, pool_size=2):
        super(CNN_Transformer, self).__init__()
        self.activation_fn = get_activation(activation)
        self.use_batch_norm = use_batch_norm
        
        num_cnn_layers = max(len(cnn_filters), len(cnn_kernel_sizes))
        if len(cnn_filters) < num_cnn_layers:
            cnn_filters = cnn_filters + [cnn_filters[-1]] * (num_cnn_layers - len(cnn_filters))
        if len(cnn_kernel_sizes) < num_cnn_layers:
            cnn_kernel_sizes = cnn_kernel_sizes + [cnn_kernel_sizes[-1]] * (num_cnn_layers - len(cnn_kernel_sizes))
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = input_channels
        current_length = sequence_length
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            padding = kernel_size // 2
            if i == len(cnn_filters) - 1:
                out_channels = d_model
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            self.dropout_layers.append(nn.Dropout(dropout_conv))
            in_channels = out_channels
            current_length = current_length // pool_size
        
        self.cnn_output_length = current_length
        
        self.pos_encoder = PositionalEncoding(d_model, dropout_transformer, max_len=self.cnn_output_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_transformer,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        fc_input_size = d_model
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            self.fc_dropouts.append(nn.Dropout(dropout_fc))
            fc_input_size = fc_size
        
        self.fc_out = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation_fn(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        for fc_layer, dropout in zip(self.fc_layers, self.fc_dropouts):
            x = self.activation_fn(fc_layer(x))
            x = dropout(x)
        x = self.fc_out(x)
        return x

# Model class map
model_class_map = {
    'CNN1D': CNN1D,
    'CNN_LSTM': CNN_LSTM,
    'CNN_Transformer': CNN_Transformer
}

@st.cache_resource
def load_model_cached(model_path, model_class_map):
    """Load a saved PyTorch model with hyperparameters and preprocessing (cached)"""
    if not os.path.exists(model_path):
        return None, None, None, None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_class_name = checkpoint['model_class']
    model_class = model_class_map[model_class_name]
    
    # Get hyperparameters from checkpoint (if saved) or use defaults
    hyperparams = checkpoint.get('hyperparameters', {})
    
    # Load preprocessing configuration and scaler
    preprocessing_config = checkpoint.get('preprocessing_config', {})
    scaler = None
    scaler_path = checkpoint.get('scaler_path')
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
    
    # Default hyperparameters for each model type
    if model_class_name == 'CNN1D':
        default_params = {
            'filters': [64, 128, 256],
            'kernel_sizes': [5, 5, 3],
            'fc_sizes': [128, 64],
            'dropout_conv': 0.3,
            'dropout_fc': 0.5,
            'activation': 'relu',
            'use_batch_norm': True,
            'pool_size': 2
        }
        default_params.update(hyperparams)
        model = model_class(
            input_channels=checkpoint['input_channels'],
            num_classes=checkpoint['num_classes'],
            sequence_length=checkpoint['sequence_length'],
            **default_params
        )
    elif model_class_name == 'CNN_LSTM':
        default_params = {
            'cnn_filters': [64, 128],
            'cnn_kernel_sizes': [5, 5],
            'lstm_hidden': 128,
            'lstm_layers': 2,
            'lstm_bidirectional': True,
            'fc_sizes': [128, 64],
            'dropout_conv': 0.3,
            'dropout_lstm': 0.3,
            'dropout_fc': 0.5,
            'activation': 'relu',
            'use_batch_norm': True,
            'pool_size': 2
        }
        default_params.update(hyperparams)
        model = model_class(
            input_channels=checkpoint['input_channels'],
            num_classes=checkpoint['num_classes'],
            sequence_length=checkpoint['sequence_length'],
            **default_params
        )
    elif model_class_name == 'CNN_Transformer':
        default_params = {
            'cnn_filters': [64, 128],
            'cnn_kernel_sizes': [5, 5],
            'd_model': 128,
            'nhead': 8,
            'num_layers': 2,
            'dim_feedforward': 512,
            'fc_sizes': [128, 64],
            'dropout_conv': 0.3,
            'dropout_transformer': 0.1,
            'dropout_fc': 0.5,
            'activation': 'relu',
            'use_batch_norm': True,
            'pool_size': 2
        }
        default_params.update(hyperparams)
        model = model_class(
            input_channels=checkpoint['input_channels'],
            num_classes=checkpoint['num_classes'],
            sequence_length=checkpoint['sequence_length'],
            **default_params
        )
    
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if model_class_name in ('CNN_Transformer', 'CNN_LSTM', 'CNN1D'):
            legacy_map = {
                'conv1': 'conv_layers.0',
                'bn1': 'bn_layers.0',
                'conv2': 'conv_layers.1',
                'bn2': 'bn_layers.1',
                'conv3': 'conv_layers.2',
                'bn3': 'bn_layers.2',
                'fc1': 'fc_layers.0',
                'fc2': 'fc_layers.1',
                'fc3': 'fc_out'
            }
            remapped_state = {}
            for key, value in state_dict.items():
                if key.endswith('num_batches_tracked'):
                    continue
                new_key = key
                for old_prefix, new_prefix in legacy_map.items():
                    if key.startswith(old_prefix + '.'):
                        new_key = key.replace(old_prefix, new_prefix, 1)
                        break
                remapped_state[new_key] = value
            model.load_state_dict(remapped_state)
        else:
            raise
    model.eval()
    
    return model, checkpoint, preprocessing_config, scaler

def preprocess_for_inference(sensor_data, preprocessing_config, scaler=None):
    """
    Preprocess sensor data for inference using saved preprocessing configuration.
    
    Args:
        sensor_data: Raw sensor data (numpy array)
        preprocessing_config: Dictionary with preprocessing settings
        scaler: Fitted scaler object (if normalization was used)
    
    Returns:
        Preprocessed sensor data ready for model input
    """
    sensor_data = sensor_data.copy()
    sequence_length = preprocessing_config.get('sequence_length', 100)
    
    # Apply drift compensation if it was used during training
    if preprocessing_config.get('handle_drift', False):
        drift_method = preprocessing_config.get('drift_method', 'baseline')
        
        if drift_method == 'baseline':
            baseline = sensor_data[0, :] if sensor_data.ndim == 2 else sensor_data[0]
            sensor_data = sensor_data - baseline
        elif drift_method == 'relative':
            baseline = sensor_data[0, :] if sensor_data.ndim == 2 else sensor_data[0]
            baseline = np.where(baseline == 0, 1, baseline)
            if sensor_data.ndim == 2:
                sensor_data = (sensor_data - baseline) / baseline
            else:
                sensor_data = (sensor_data - baseline) / baseline
        elif drift_method == 'moving_baseline':
            window_size = min(10, sequence_length // 10)
            if sensor_data.ndim == 2:
                baseline = np.mean(sensor_data[:window_size, :], axis=0)
            else:
                baseline = np.mean(sensor_data[:window_size])
            sensor_data = sensor_data - baseline
    
    # Apply normalization if it was used during training
    if scaler is not None and preprocessing_config.get('normalization', 'none') != 'none':
        normalization = preprocessing_config.get('normalization', 'standard')
        
        if normalization == 'per_sensor':
            # Per-sensor normalization
            if sensor_data.ndim == 2:
                n_samples, n_features = sensor_data.shape
                sensor_scaled = np.zeros_like(sensor_data)
                for sensor_idx in range(n_features):
                    sensor_scaled[:, sensor_idx] = scaler[sensor_idx].transform(
                        sensor_data[:, sensor_idx].reshape(-1, 1)
                    ).flatten()
                sensor_data = sensor_scaled
            else:
                # Single sensor
                sensor_data = scaler[0].transform(sensor_data.reshape(-1, 1)).flatten()
        else:
            # Global normalization
            if sensor_data.ndim == 2:
                sensor_data = scaler.transform(sensor_data)
            else:
                sensor_data = scaler.transform(sensor_data.reshape(-1, 1)).flatten()
    
    # Ensure correct shape and length
    if sensor_data.ndim == 1:
        # Single row, pad or truncate to sequence_length
        if len(sensor_data) < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - len(sensor_data),))
            sensor_data = np.concatenate([sensor_data, padding])
        else:
            sensor_data = sensor_data[:sequence_length]
        # Reshape to (1, sequence_length, num_features)
        num_features = preprocessing_config.get('num_features', sensor_data.shape[-1])
        sensor_data = sensor_data.reshape(1, sequence_length, num_features)
    elif sensor_data.ndim == 2:
        # 2D array: (time_steps, features)
        if sensor_data.shape[0] < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - sensor_data.shape[0], sensor_data.shape[1]))
            sensor_data = np.vstack([sensor_data, padding])
        else:
            sensor_data = sensor_data[:sequence_length]
        # Reshape to (1, sequence_length, num_features)
        sensor_data = sensor_data.reshape(1, sequence_length, -1)
    
    return sensor_data.astype(np.float32)

def predict_freshness(model, sensor_data, device, class_names, 
                     preprocessing_config=None, scaler=None, sequence_length=100):
    """
    Predict fruit freshness from sensor data with proper preprocessing.
    
    Args:
        model: Trained PyTorch model
        sensor_data: Raw sensor readings (numpy array)
        device: Device to run inference on
        class_names: List of class names
        preprocessing_config: Dictionary with preprocessing settings (from saved model)
        scaler: Fitted scaler object (from saved model)
        sequence_length: Expected sequence length (fallback if preprocessing_config not provided)
    
    Returns:
        predicted_class: Predicted class name
        probabilities: Dictionary of class probabilities
    """
    model.eval()
    
    # Use preprocessing_config if provided, otherwise fall back to simple padding
    if preprocessing_config is not None:
        # Use saved preprocessing configuration
        preprocessed_data = preprocess_for_inference(sensor_data, preprocessing_config, scaler)
    else:
        # Legacy mode: simple padding/truncation (no normalization or drift compensation)
        if sensor_data.ndim == 1:
            if len(sensor_data) < sequence_length:
                sensor_data = np.pad(sensor_data, (0, sequence_length - len(sensor_data)), 
                                   mode='constant', constant_values=0)
            else:
                sensor_data = sensor_data[:sequence_length]
            sensor_data = sensor_data.reshape(1, sequence_length, -1)
        elif sensor_data.ndim == 2:
            if sensor_data.shape[0] < sequence_length:
                padding = np.zeros((sequence_length - sensor_data.shape[0], sensor_data.shape[1]))
                sensor_data = np.vstack([sensor_data, padding])
            else:
                sensor_data = sensor_data[:sequence_length]
            sensor_data = sensor_data.reshape(1, sequence_length, -1)
        preprocessed_data = sensor_data.astype(np.float32)
    
    sensor_tensor = torch.FloatTensor(preprocessed_data).to(device)
    
    with torch.no_grad():
        outputs = model(sensor_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    prob_dict = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
    
    return class_names[predicted_idx], prob_dict

def create_fancy_confusion_matrix(y_true, y_pred, class_names, title):
    """Create an interactive confusion matrix with Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        xaxis=dict(title=dict(text="Predicted", font=dict(size=14))),
        yaxis=dict(title=dict(text="Actual", font=dict(size=14))),
        width=700,
        height=600
    )
    
    return fig

def create_training_curves(history, title):
    """Create interactive training curves with Plotly"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss', 'Accuracy'),
        vertical_spacing=0.15
    )
    
    # Loss curves
    fig.add_trace(
        go.Scatter(x=list(range(len(history['train_loss']))), 
                  y=history['train_loss'], 
                  mode='lines+markers',
                  name='Train Loss',
                  line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history['val_loss']))), 
                  y=history['val_loss'], 
                  mode='lines+markers',
                  name='Val Loss',
                  line=dict(color='#f093fb', width=2)),
        row=1, col=1
    )
    
    # Accuracy curves
    fig.add_trace(
        go.Scatter(x=list(range(len(history['train_acc']))), 
                  y=history['train_acc'], 
                  mode='lines+markers',
                  name='Train Acc',
                  line=dict(color='#4facfe', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history['val_acc']))), 
                  y=history['val_acc'], 
                  mode='lines+markers',
                  name='Val Acc',
                  line=dict(color='#00f2fe', width=2)),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#667eea')),
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def load_all_models():
    """Load all PyTorch models with preprocessing configs"""
    models_dir = 'saved_models_pytorch'
    loaded_models = {}
    model_metadata = {}
    preprocessing_configs = {}
    scalers = {}
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model, metadata, prep_config, scaler = load_model_cached(model_path, model_class_map)
            if model is not None:
                config_name = metadata.get('config_name', model_file.replace('.pth', ''))
                loaded_models[config_name] = model
                model_metadata[config_name] = metadata
                preprocessing_configs[config_name] = prep_config
                scalers[config_name] = scaler
    
    return loaded_models, model_metadata, preprocessing_configs, scalers

def format_config_label(config_name, metadata, preprocessing_config):
    """Create a human-friendly label for a configuration."""
    model_class = metadata.get('model_class', 'Unknown')
    hyperparams = metadata.get('hyperparameters', {}) if metadata else {}
    prep_name = preprocessing_config.get('name') if preprocessing_config else None

    parts = []
    if prep_name:
        parts.append(prep_name)
    parts.append(model_class)

    if model_class == 'CNN1D':
        filters = hyperparams.get('filters', [64, 128, 256])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"conv={len(filters)}")
        parts.append(f"fc={len(fc_sizes)}")
        parts.append(f"act={hyperparams.get('activation', 'relu')}")
    elif model_class == 'CNN_LSTM':
        cnn_filters = hyperparams.get('cnn_filters', [64, 128])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"conv={len(cnn_filters)}")
        parts.append(f"lstm={hyperparams.get('lstm_layers', 2)}")
        parts.append(f"fc={len(fc_sizes)}")
        parts.append(f"act={hyperparams.get('activation', 'relu')}")
    elif model_class == 'CNN_Transformer':
        cnn_filters = hyperparams.get('cnn_filters', [64, 128])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"conv={len(cnn_filters)}")
        parts.append(f"layers={hyperparams.get('num_layers', 2)}")
        parts.append(f"heads={hyperparams.get('nhead', 8)}")
        parts.append(f"fc={len(fc_sizes)}")
        parts.append(f"act={hyperparams.get('activation', 'relu')}")

    label = " | ".join(parts)
    return f"{label} ({config_name})"

def format_config_summary(metadata, preprocessing_config):
    """Create a concise, intuitive description without the config id."""
    model_class = metadata.get('model_class', 'Unknown')
    hyperparams = metadata.get('hyperparameters', {}) if metadata else {}
    prep_name = preprocessing_config.get('name') if preprocessing_config else None

    parts = []
    if prep_name:
        parts.append(prep_name)
    parts.append(model_class)

    if model_class == 'CNN1D':
        filters = hyperparams.get('filters', [64, 128, 256])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"Conv {len(filters)}")
        parts.append(f"FC {len(fc_sizes)}")
        parts.append(f"Act {hyperparams.get('activation', 'relu')}")
    elif model_class == 'CNN_LSTM':
        cnn_filters = hyperparams.get('cnn_filters', [64, 128])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"Conv {len(cnn_filters)}")
        parts.append(f"LSTM {hyperparams.get('lstm_layers', 2)}")
        parts.append(f"FC {len(fc_sizes)}")
        parts.append(f"Act {hyperparams.get('activation', 'relu')}")
    elif model_class == 'CNN_Transformer':
        cnn_filters = hyperparams.get('cnn_filters', [64, 128])
        fc_sizes = hyperparams.get('fc_sizes', [128, 64])
        parts.append(f"Conv {len(cnn_filters)}")
        parts.append(f"Layers {hyperparams.get('num_layers', 2)}")
        parts.append(f"Heads {hyperparams.get('nhead', 8)}")
        parts.append(f"FC {len(fc_sizes)}")
        parts.append(f"Act {hyperparams.get('activation', 'relu')}")

    return " • ".join(parts)

def get_config_info(config_name, metadata, preprocessing_config):
    """Extract filterable attributes for a configuration."""
    model_class = metadata.get('model_class', 'Unknown')
    hyperparams = metadata.get('hyperparameters', {}) if metadata else {}
    prep_name = preprocessing_config.get('name') if preprocessing_config else None

    if model_class == 'CNN1D':
        conv_layers = len(hyperparams.get('filters', [64, 128, 256]))
    else:
        conv_layers = len(hyperparams.get('cnn_filters', [64, 128]))

    info = {
        'config_name': config_name,
        'model_class': model_class,
        'preprocessing': prep_name,
        'preprocessing_config': preprocessing_config or {},
        'activation': hyperparams.get('activation', 'relu'),
        'conv_layers': conv_layers,
        'fc_layers': len(hyperparams.get('fc_sizes', [128, 64]))
    }

    if model_class == 'CNN_LSTM':
        info['lstm_layers'] = hyperparams.get('lstm_layers', 2)
    if model_class == 'CNN_Transformer':
        info['transformer_layers'] = hyperparams.get('num_layers', 2)
        info['transformer_heads'] = hyperparams.get('nhead', 8)

    return info

def parse_classification_report_text(report_text):
    """Parse a sklearn-style classification report text."""
    lines = [line.rstrip() for line in report_text.splitlines()]
    header_idx = None
    for i, line in enumerate(lines):
        if "precision" in line and "recall" in line and "f1-score" in line:
            header_idx = i
            break
    table_lines = []
    if header_idx is not None:
        for line in lines[header_idx:]:
            if line.strip().startswith("accuracy") or line.strip().startswith("Summary Metrics"):
                break
            if line.strip():
                table_lines.append(line)
    table_df = None
    if table_lines:
        try:
            table_df = pd.read_fwf(io.StringIO("\n".join(table_lines)), index_col=0)
        except Exception:
            table_df = None

    summary_metrics = {}
    for line in lines:
        match = re.match(r"^([A-Za-z\s\(\)\-]+):\s*([0-9\.]+)\s*$", line.strip())
        if match:
            key = match.group(1).strip()
            value = float(match.group(2))
            summary_metrics[key] = value
    return table_df, summary_metrics

def render_report_summary(summary_metrics):
    """Render summary metrics in a compact layout."""
    if not summary_metrics:
        return
    cols = st.columns(4)
    metric_items = list(summary_metrics.items())
    for idx, (key, value) in enumerate(metric_items[:4]):
        cols[idx].metric(key, f"{value:.4f}")
    if len(metric_items) > 4:
        cols = st.columns(4)
        for idx, (key, value) in enumerate(metric_items[4:8]):
            cols[idx].metric(key, f"{value:.4f}")

def render_report_from_dict(report_dict):
    """Render a classification report dict with summary stats."""
    summary_metrics = {}
    if "accuracy" in report_dict:
        summary_metrics["Accuracy"] = report_dict["accuracy"]
    if "macro avg" in report_dict:
        summary_metrics["Precision (macro)"] = report_dict["macro avg"].get("precision", 0)
        summary_metrics["Recall (macro)"] = report_dict["macro avg"].get("recall", 0)
        summary_metrics["F1-Score (macro)"] = report_dict["macro avg"].get("f1-score", 0)
    if "weighted avg" in report_dict:
        summary_metrics["Precision (weighted)"] = report_dict["weighted avg"].get("precision", 0)
        summary_metrics["Recall (weighted)"] = report_dict["weighted avg"].get("recall", 0)
        summary_metrics["F1-Score (weighted)"] = report_dict["weighted avg"].get("f1-score", 0)
    render_report_summary(summary_metrics)
    report_df = pd.DataFrame(report_dict).transpose()
    with st.expander("Show detailed per-class report", expanded=False):
        st.dataframe(report_df, use_container_width=True)

def make_quality_control_decision(predicted_class, probabilities, confidence_threshold=0.7):
    """Make a quality control decision from prediction output."""
    max_prob = max(probabilities.values())
    match = re.search(r"\d+", str(predicted_class))
    freshness_day = int(match.group()) if match else 0

    if max_prob < confidence_threshold:
        decision = "UNCERTAIN"
        action = "Manual inspection required"
        reason = f"Low confidence ({max_prob:.2f})"
    elif freshness_day <= 2:
        decision = "FRESH"
        action = "Approve for sale/storage"
        reason = f"Freshness stage: {predicted_class} (Days 1-2)"
    elif freshness_day == 3:
        decision = "MODERATE"
        action = "Prioritize sale, monitor closely"
        reason = f"Freshness stage: {predicted_class} (Day 3)"
    else:
        decision = "AT_RISK"
        action = "Immediate sale or discard"
        reason = f"Freshness stage: {predicted_class} (Days 4-5)"

    return {
        'predicted_class': predicted_class,
        'confidence': max_prob,
        'decision': decision,
        'action': action,
        'reason': reason,
        'all_probabilities': probabilities
    }

def compute_metrics_from_labels(y_true, y_pred):
    """Compute test metrics from labels/predictions."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

@st.cache_data
def load_sequences_from_dir(data_dir, sequence_length, window_stride):
    """Load raw sequences and labels from CSV files in data_dir."""
    freshness_mapping = {'D1': 0, 'D2': 1, 'D3': 2, 'D4': 3, 'D5': 4}
    freshness_labels = {v: k for k, v in freshness_mapping.items()}

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    stride = window_stride if window_stride is not None else sequence_length
    all_sequences = []
    all_labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            parts = filename.replace('.csv', '').split(' ')
            if len(parts) > 1 and parts[-1] in freshness_mapping:
                label_str = parts[-1]
                label = freshness_mapping[label_str]
                df = pd.read_csv(filepath)
                sensor_columns = [col for col in df.columns if col.startswith('MQ')]
                if not sensor_columns:
                    continue
                sensor_data = df[sensor_columns].values
                if len(sensor_data) >= sequence_length:
                    for i in range(0, len(sensor_data) - sequence_length + 1, stride):
                        sequence = sensor_data[i:i + sequence_length]
                        all_sequences.append(sequence)
                        all_labels.append(label)
                else:
                    padded = np.pad(
                        sensor_data,
                        ((0, sequence_length - len(sensor_data)), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                    all_sequences.append(padded)
                    all_labels.append(label)

    if not all_sequences:
        raise ValueError("No valid data sequences found in the data directory.")

    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int64), freshness_labels

def select_configuration_with_filters(config_info_list, key_prefix="", show_preprocessing_help=True):
    """Render cascading select boxes and return chosen configuration."""
    filtered = list(config_info_list)

    def options_from(field):
        return sorted({c.get(field) for c in filtered if c.get(field) is not None})

    def preprocessing_description(prep_name, prep_config):
        norm = prep_config.get('normalization', 'none')
        window_stride = prep_config.get('window_stride')
        handle_drift = prep_config.get('handle_drift', False)
        drift_method = prep_config.get('drift_method')
        sequence_length = prep_config.get('sequence_length')
        num_features = prep_config.get('num_features')

        norm_text = {
            'none': "Uses raw sensor values without scaling",
            'standard': "Balances values so no sensor dominates (mean 0, std 1)",
            'minmax': "Scales values into a 0–1 range",
            'robust': "Scales using median/IQR to reduce outlier impact",
            'per_sensor': "Scales each sensor independently"
        }.get(norm, f"Uses `{norm}` scaling")

        if window_stride in (None, 0, 100):
            window_text = "Uses non-overlapping windows"
        elif window_stride == 1:
            window_text = "Uses overlapping windows to create more samples (stride=1)"
        else:
            window_text = f"Uses partially overlapping windows (stride={window_stride})"

        if not handle_drift:
            drift_text = "No drift correction"
        else:
            drift_text = {
                'baseline': "Drift correction by subtracting the first value",
                'relative': "Drift correction using relative change from the first value",
                'moving_baseline': "Drift correction using a moving average baseline"
            }.get(drift_method, "Drift correction enabled")

        details = [norm_text, window_text, drift_text]
        if sequence_length is not None:
            details.append(f"Sequence length: {sequence_length}")
        if num_features is not None:
            details.append(f"Sensors used: {num_features}")
        return f"**{prep_name}** — " + "; ".join(details)

    preprocessing_options = options_from('preprocessing')
    selected_preprocessing = st.selectbox(
        "Preprocessing",
        ["All"] + preprocessing_options,
        key=f"{key_prefix}_prep"
    )
    if show_preprocessing_help:
        if selected_preprocessing != "All":
            matching = [
                c for c in config_info_list
                if c.get('preprocessing') == selected_preprocessing
            ]
            config = matching[0].get('preprocessing_config', {}) if matching else {}
            st.info(preprocessing_description(selected_preprocessing, config))
        else:
            with st.expander("What does each preprocessing mean?", expanded=False):
                if preprocessing_options:
                    for prep_name in preprocessing_options:
                        matching = [
                            c for c in config_info_list
                            if c.get('preprocessing') == prep_name
                        ]
                        config = matching[0].get('preprocessing_config', {}) if matching else {}
                        st.markdown(preprocessing_description(prep_name, config))
                else:
                    st.info("No preprocessing configurations found.")
    if selected_preprocessing != "All":
        filtered = [c for c in filtered if c.get('preprocessing') == selected_preprocessing]

    model_type_options = [m for m in options_from('model_class') if m != "CNN_Transformer"]
    selected_model_type = st.selectbox(
        "Model Type",
        ["All"] + model_type_options,
        key=f"{key_prefix}_model_type"
    )
    if selected_model_type != "All":
        filtered = [c for c in filtered if c.get('model_class') == selected_model_type]
    else:
        filtered = [c for c in filtered if c.get('model_class') != "CNN_Transformer"]

    activation_options = options_from('activation')
    selected_activation = st.selectbox(
        "Activation",
        ["All"] + activation_options,
        key=f"{key_prefix}_activation"
    )
    if selected_activation != "All":
        filtered = [c for c in filtered if c.get('activation') == selected_activation]

    conv_layer_options = options_from('conv_layers')
    selected_conv_layers = st.selectbox(
        "Conv Layers",
        ["All"] + conv_layer_options,
        key=f"{key_prefix}_conv_layers"
    )
    if selected_conv_layers != "All":
        filtered = [c for c in filtered if c.get('conv_layers') == selected_conv_layers]

    fc_layer_options = options_from('fc_layers')
    selected_fc_layers = st.selectbox(
        "FC Layers",
        ["All"] + fc_layer_options,
        key=f"{key_prefix}_fc_layers"
    )
    if selected_fc_layers != "All":
        filtered = [c for c in filtered if c.get('fc_layers') == selected_fc_layers]

    if selected_model_type == "CNN_LSTM":
        lstm_layer_options = options_from('lstm_layers')
        selected_lstm_layers = st.selectbox(
            "LSTM Layers",
            ["All"] + lstm_layer_options,
            key=f"{key_prefix}_lstm_layers"
        )
        if selected_lstm_layers != "All":
            filtered = [c for c in filtered if c.get('lstm_layers') == selected_lstm_layers]

    if selected_model_type == "CNN_Transformer":
        transformer_layer_options = options_from('transformer_layers')
        transformer_head_options = options_from('transformer_heads')
        selected_transformer_layers = st.selectbox(
            "Transformer Layers",
            ["All"] + transformer_layer_options,
            key=f"{key_prefix}_transformer_layers"
        )
        selected_transformer_heads = st.selectbox(
            "Transformer Heads",
            ["All"] + transformer_head_options,
            key=f"{key_prefix}_transformer_heads"
        )
        if selected_transformer_layers != "All":
            filtered = [c for c in filtered if c.get('transformer_layers') == selected_transformer_layers]
        if selected_transformer_heads != "All":
            filtered = [c for c in filtered if c.get('transformer_heads') == selected_transformer_heads]

    available_configs = [c.get('config_name') for c in filtered]
    if not available_configs:
        st.warning("No configurations match the selected filters.")
        st.stop()

    return available_configs

@st.cache_data
def load_training_results():
    """Load training results for all configurations (cached)."""
    results_paths = [
        'training_results_all_configs.pkl',
        'training_results.pkl'
    ]
    for results_path in results_paths:
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f) or {}
                if results:
                    return results
    return {}

# Initialize session state
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models, st.session_state.model_metadata, \
    st.session_state.preprocessing_configs, st.session_state.scalers = load_all_models()
if 'training_results' not in st.session_state:
    st.session_state.training_results = load_training_results()

# Sidebar navigation
st.sidebar.title("🍎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Model Comparison", "🔍 Training Analysis", "🔮 Make Prediction"])

# Home page
if page == "🏠 Home":
    st.markdown("""
        <div class="main-header">
            <h1>🍎 Fruit Quality Monitoring</h1>
            <p>Deep Learning Dashboard - Compare Multiple Model Configurations</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3>📋 About This Dashboard</h3>
            <p>This dashboard loads and compares <strong>pre-trained</strong> model configurations with different hyperparameters for educational purposes.</p>
            <p><strong>Note:</strong> Models were pre-trained in advance, not using this demo app. This dashboard is for visualization and inference only.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📘 Configuration & Analysis Guide", expanded=False):
        st.markdown("""
        **Preprocessing options (how data is prepared)**
        - **Normalization**: `none` (raw values), `standard` (mean 0, std 1), `minmax` (0–1 range), `robust` (median/IQR), `per_sensor` (each sensor scaled separately).
        - **Windowing**: non‑overlapping windows (default), sliding window (`stride=1`), or custom stride (`stride=N`).
        - **Drift handling**: `baseline` (subtract first value), `relative` (change relative to baseline), `moving_baseline` (moving average baseline).

        **Model configurations (what gets trained)**
        - **CNN1D**: convolutional layers extract features, followed by fully‑connected layers.
        - **CNN_LSTM**: convolutional layers + LSTM for temporal patterns.
        - Hyperparameters shown in labels (e.g., conv layers, LSTM layers, FC layers, activation).

        **How to interpret analysis**
        - **Training curves**: loss/accuracy over epochs (watch for overfitting).
        - **Confusion matrices**: where the model confuses freshness stages.
        - **Classification reports**: per‑class precision/recall/F1 and overall averages.
        - **Model comparison**: uses **test‑set metrics only** for fair comparison.
        """)
    
    if st.session_state.loaded_models:
        visible_configs = [
            k for k in st.session_state.loaded_models.keys()
            if st.session_state.model_metadata.get(k, {}).get('model_class') != "CNN_Transformer"
        ]
        st.success(f"✅ Loaded {len(visible_configs)} model configurations")
        
        st.markdown("### 📦 Available Configurations")
        config_df = pd.DataFrame({
            'Configuration': visible_configs,
            'Description': [
                format_config_label(
                    k,
                    st.session_state.model_metadata.get(k, {}),
                    st.session_state.preprocessing_configs.get(k, {})
                )
                for k in visible_configs
            ],
            'Model Type': [st.session_state.model_metadata[k].get('model_class', 'Unknown') 
                          for k in visible_configs]
        })
        st.dataframe(config_df, use_container_width=True)

        # Training results table removed from Home page
    else:
        st.warning("⚠️ No model configurations found in 'saved_models_pytorch/' directory.")
        st.info("💡 Please train models in the Jupyter notebook first and save them.")

# Model Comparison page
elif page == "📊 Model Comparison":
    st.markdown("<h1 style='color: #667eea;'>📊 Model Comparison</h1>", unsafe_allow_html=True)
    
    training_results = st.session_state.training_results
    if training_results:
            # Prepare comparison data (test metrics only)
            comparison_data = []
            available_configs = [
                k for k in st.session_state.loaded_models.keys()
                if st.session_state.model_metadata.get(k, {}).get('model_class') != "CNN_Transformer"
            ]
            for config_name in available_configs:
                results = training_results.get(config_name, {})
                metrics = results.get('test_metrics', {})
                if not metrics and 'test_labels' in results and 'test_predictions' in results:
                    metrics = compute_metrics_from_labels(
                        results['test_labels'],
                        results['test_predictions']
                    )
                if metrics:
                    comparison_data.append({
                        'Configuration': config_name,
                        'Accuracy (Test)': metrics.get('accuracy', 0) * 100,
                        'Precision (Macro)': metrics.get('precision_macro', 0) * 100,
                        'Recall (Macro)': metrics.get('recall_macro', 0) * 100,
                        'F1-Score (Macro)': metrics.get('f1_macro', 0) * 100,
                        'F1-Score (Weighted)': metrics.get('f1_weighted', 0) * 100
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                label_map = {
                    k: format_config_label(
                        k,
                        st.session_state.model_metadata.get(k, {}),
                        st.session_state.preprocessing_configs.get(k, {})
                    )
                    for k in available_configs
                }
                comparison_df.insert(
                    0,
                    'Label',
                    comparison_df['Configuration'].map(lambda k: label_map.get(k, k))
                )

                metric_options = [
                    'Accuracy (Test)',
                    'Precision (Macro)',
                    'Recall (Macro)',
                    'F1-Score (Macro)',
                    'F1-Score (Weighted)'
                ]
                selected_metric = st.selectbox("Metric to compare", metric_options, index=0)
                plot_df = comparison_df.sort_values(selected_metric, ascending=True)

                fig = px.bar(
                    plot_df,
                    x=selected_metric,
                    y='Label',
                    orientation='h',
                    title=f"Model Comparison — {selected_metric}",
                    color=selected_metric,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_title='Score (%)',
                    yaxis_title='Configuration',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.markdown("### 📋 Detailed Metrics")
                st.dataframe(comparison_df.drop(columns=['Configuration']), use_container_width=True)
            else:
                st.warning("No test metrics found in training results.")
    else:
        st.warning("⚠️ No training results found. Please train models in the notebook first.")

# Training Analysis page
elif page == "🔍 Training Analysis":
    st.markdown("<h1 style='color: #667eea;'>🔍 Training Analysis</h1>", unsafe_allow_html=True)
    
    if st.session_state.loaded_models:
        config_info_list = [
            get_config_info(
                k,
                st.session_state.model_metadata.get(k, {}),
                st.session_state.preprocessing_configs.get(k, {})
            )
            for k in st.session_state.loaded_models.keys()
        ]
        available_configs = select_configuration_with_filters(
            config_info_list,
            key_prefix="training_analysis"
        )

        label_map = {
            k: format_config_label(
                k,
                st.session_state.model_metadata.get(k, {}),
                st.session_state.preprocessing_configs.get(k, {})
            )
            for k in available_configs
        }
        selected_config = st.selectbox(
            "Configuration",
            available_configs,
            format_func=lambda k: label_map.get(k, k)
        )
        
        training_results = st.session_state.training_results
        if selected_config in training_results:
            results = training_results[selected_config]
            
            # Training curves
            if 'history' in results:
                st.markdown("### 📈 Training Curves")
                fig = create_training_curves(results['history'], f"Training History - {selected_config}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification reports
            if 'val_labels' in results and 'val_predictions' in results:
                st.markdown("### 📋 Validation Classification Report")
                class_names = results.get('class_names', ['D1', 'D2', 'D3', 'D4', 'D5'])
                val_report = classification_report(
                    results['val_labels'],
                    results['val_predictions'],
                    target_names=class_names,
                    output_dict=True
                )
                render_report_from_dict(val_report)

                st.markdown("### 📊 Confusion Matrix (Validation Set)")
                val_cm_fig = create_fancy_confusion_matrix(
                    results['val_labels'],
                    results['val_predictions'],
                    class_names,
                    f"Validation Confusion Matrix - {selected_config}"
                )
                st.plotly_chart(val_cm_fig, use_container_width=True)

            if 'test_labels' in results and 'test_predictions' in results:
                st.markdown("### 📋 Test Classification Report")
                class_names = results.get('class_names', ['D1', 'D2', 'D3', 'D4', 'D5'])
                test_report = classification_report(
                    results['test_labels'], 
                    results['test_predictions'],
                    target_names=class_names,
                    output_dict=True
                )
                render_report_from_dict(test_report)

                st.markdown("### 📊 Confusion Matrix (Test Set)")
                test_cm_fig = create_fancy_confusion_matrix(
                    results['test_labels'],
                    results['test_predictions'],
                    class_names,
                    f"Test Confusion Matrix - {selected_config}"
                )
                st.plotly_chart(test_cm_fig, use_container_width=True)
        else:
            st.warning(f"No training results found for {selected_config}")
    else:
        st.warning("⚠️ No models loaded. Please train models in the notebook first.")

# Make Prediction page
elif page == "🔮 Make Prediction":
    st.markdown("<h1 style='color: #667eea;'>🔮 Make Prediction</h1>", unsafe_allow_html=True)
    
    if st.session_state.loaded_models:
        config_info_list = [
            get_config_info(
                k,
                st.session_state.model_metadata.get(k, {}),
                st.session_state.preprocessing_configs.get(k, {})
            )
            for k in st.session_state.loaded_models.keys()
        ]
        available_configs = select_configuration_with_filters(
            config_info_list,
            key_prefix="prediction",
            show_preprocessing_help=False
        )

        label_map = {
            k: format_config_label(
                k,
                st.session_state.model_metadata.get(k, {}),
                st.session_state.preprocessing_configs.get(k, {})
            )
            for k in available_configs
        }
        selected_config = st.selectbox(
            "Configuration",
            available_configs,
            format_func=lambda k: label_map.get(k, k)
        )
        
        st.markdown("### 📦 Automatic Dataset (No Upload Needed)")
        data_dir = "./AllSmaples-Report"
        st.info(f"Using data from `{data_dir}`")

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.5,
            max_value=0.99,
            value=0.7,
            step=0.01
        )

        model = st.session_state.loaded_models[selected_config]
        metadata = st.session_state.model_metadata[selected_config]
        class_names = metadata.get('class_names', ['D1', 'D2', 'D3', 'D4', 'D5'])
        device = torch.device('cpu')
        prep_config = st.session_state.preprocessing_configs.get(selected_config, {})
        scaler = st.session_state.scalers.get(selected_config)

        sequence_length = prep_config.get('sequence_length', 100)
        window_stride = prep_config.get('window_stride')

        try:
            sequences, labels, freshness_labels = load_sequences_from_dir(
                data_dir, sequence_length, window_stride
            )

            _, test_indices = train_test_split(
                range(len(sequences)),
                test_size=0.25,
                random_state=42,
                stratify=labels
            )
            test_sequences = sequences[test_indices]
            test_labels = labels[test_indices]

            st.markdown("### ✅ Quality Control Decision Examples")
            max_examples = min(10, len(test_sequences))
            num_examples = st.slider(
                "Number of examples",
                min_value=1,
                max_value=max_examples,
                value=min(5, max_examples)
            )

            if st.button("Generate Decision Examples", type="primary"):
                rng = np.random.default_rng(42)
                selected_indices = rng.choice(
                    range(len(test_sequences)),
                    size=num_examples,
                    replace=False
                )

                output_lines = [
                    "Quality Control Decision Examples:",
                    "=" * 60
                ]
                for i, idx in enumerate(selected_indices, start=1):
                    sample_sequence = test_sequences[idx]
                    true_label = freshness_labels.get(int(test_labels[idx]), f"D{int(test_labels[idx]) + 1}")

                    predicted_class, probabilities = predict_freshness(
                        model, sample_sequence, device, class_names,
                        preprocessing_config=prep_config,
                        scaler=scaler
                    )
                    decision = make_quality_control_decision(
                        predicted_class, probabilities, confidence_threshold
                    )

                    output_lines.extend([
                        f"\nRandom Sample {i}:",
                        f"  True Label: {true_label}",
                        f"  Predicted: {decision['predicted_class']} (confidence: {decision['confidence']:.2f})",
                        f"  Decision: {decision['decision']}",
                        f"  Action: {decision['action']}",
                        f"  Reason: {decision['reason']}"
                    ])

                st.code("\n".join(output_lines))
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("⚠️ No models loaded. Please train models in the notebook first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🍎 Fruit Quality Monitoring Dashboard | Deep Learning Models</p>
        <p><small>Compare multiple pre-trained model configurations</small></p>
    </div>
""", unsafe_allow_html=True)
