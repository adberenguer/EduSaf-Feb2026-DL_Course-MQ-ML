import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Set page config
st.set_page_config(
    page_title="Fruit Freshness Classification | Hananto & Ridwan (2025)",
    page_icon="🍎",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Load and prepare the dataset"""
    report_dir = "./AllSmaples-Report/"
    
    # Get all CSV files that do NOT contain D1, D2, D3, D4, or D5
    all_csv_files = glob.glob(os.path.join(report_dir, "*.csv"))
    fruit_mapping = {}
    
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        if not any(f"D{i}" in filename for i in range(1, 6)):
            fruit_name = filename.replace(".csv", "")
            fruit_mapping[fruit_name] = filename
    
    # Load all mapped CSV files
    dataframes = []
    for fruit_name, filename in fruit_mapping.items():
        file_path = os.path.join(report_dir, filename)
        df = pd.read_csv(file_path)
        df['label'] = fruit_name
        dataframes.append(df)
    
    # Combine all dataframes
    full_dataframe = pd.concat(dataframes, ignore_index=True)
    
    # Shuffle
    full_dataframe = full_dataframe.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Encode labels
    label_map = {name: idx for idx, name in enumerate(sorted(full_dataframe['label'].unique()))}
    full_dataframe['label'] = full_dataframe['label'].map(label_map)
    
    # Create reverse mapping
    label_to_name = {v: k for k, v in label_map.items()}
    
    return full_dataframe, label_to_name, fruit_mapping

def define_models():
    """Define models aligned with Hananto & Ridwan (2025)"""
    models = {
        'ANN (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(max_iter=500, random_state=42))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', 'passthrough'),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    }
    return models

def train_and_evaluate(X_train, y_train, X_test, y_test, models):
    """Train models and evaluate with paper-aligned metrics"""
    trained_models = {}
    test_results = {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_models = len(models)

    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train
        trained_models[name] = model

        status_text.text(f"Testing {name}...")
        start_test = time.perf_counter()
        y_pred = model.predict(X_test)
        test_time = time.perf_counter() - start_test

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        test_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'test_time': test_time,
            'y_pred': y_pred
        }
        progress_bar.progress((idx + 1) / total_models)

    status_text.text("Training complete!")
    progress_bar.empty()
    status_text.empty()

    return trained_models, test_results

def plot_performance_comparison(test_results):
    """Plot performance comparison (accuracy, precision, recall, F1-score)."""
    import numpy as np

    models = list(test_results.keys())
    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, (title, key) in zip(axes, metrics):
        values = [test_results[m][key] for m in models]
        ax.bar(np.arange(len(models)), values, color="tab:blue", alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=25, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig

def plot_performance_comparison_legacy(test_results):
    """Plot comparison for legacy metadata (accuracy & F1-score only)."""
    import numpy as np

    models = list(test_results.keys())
    metrics = [
        ("Accuracy", "accuracy"),
        ("F1-Score", "f1_score")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes = axes.flatten()

    for ax, (title, key) in zip(axes, metrics):
        values = [test_results[m][key] for m in models]
        ax.bar(np.arange(len(models)), values, color="tab:orange", alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=25, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig

def map_model_filename_to_label(filename):
    """Map saved model filenames to display names."""
    base = os.path.basename(filename).replace(".joblib", "")
    mapping = {
        "ann_mlp": "ANN (MLP)",
        "knn": "KNN",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "svm": "SVM (Legacy)",
        "xgboost": "XGBoost (Legacy)"
    }
    return mapping.get(base, base.replace("_", " ").title())

def plot_confusion_matrix(y_test, y_pred, label_to_name, model_name):
    """Plot confusion matrix with robust handling and Streamlit support"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Prepare mapping from labels to integer indices if needed
    # Get all unique classes from both y_test and y_pred to avoid missing classes in cm display
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    # Ensure that label_to_name includes all present labels
    class_names = [label_to_name[i] if i in label_to_name else str(i) for i in unique_labels]

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=15)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()

    return fig

# Main App
st.markdown('<h1 class="main-header">🍎 Fruit Freshness Classification ML Pipeline</h1>', unsafe_allow_html=True)
st.markdown(
    "<div class='metric-card'>"
    "<b>Reference:</b> Hananto & Ridwan (2025) — Performance comparison of algorithms in the classification "
    "of fresh fruit types based on MQ array sensor data. "
    "This app mirrors the paper’s workflow: combine dataset CSVs, random train/test split, "
    "then evaluate ANN, KNN, Logistic Regression, and Random Forest using accuracy, precision, recall, and F1."
    "</div>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Train Models", "View Performance", "Make Prediction"])

# Load data (cached)
@st.cache_data
def load_cached_data():
    return load_data()

if page == "Train Models":
    st.header("Train Machine Learning Models")
    
    # List the models that will be trained
    st.markdown("**Models to be trained (paper-aligned):**")
    model_names = [
        "Artificial Neural Network (MLPClassifier)",
        "K-Nearest Neighbors (KNN)",
        "Logistic Regression (LR)",
        "Random Forest (RF)"
    ]
    st.markdown("<br>".join(model_names) + "<br>", unsafe_allow_html=True)
    
    if st.button("🔄 Train All Models", type="primary"):
        with st.spinner("Loading data..."):
            full_dataframe, label_to_name, fruit_mapping = load_cached_data()
        
        st.success(f"✓ Data loaded: {len(full_dataframe)} samples")
        
        # Prepare features
        feature_columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135']
        X = full_dataframe[feature_columns].values
        y = full_dataframe['label'].values
        
        # Random split (paper-aligned)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.info(f"📊 Dataset split: Train ({len(X_train)} samples), Test ({len(X_test)} samples)")
        
        # Define models
        models = define_models()
        
        # Train and evaluate
        with st.spinner("Training models... This may take a few minutes."):
            trained_models, test_results = train_and_evaluate(
                X_train, y_train, X_test, y_test, models
            )
        
        # Save models
        models_dir = 'saved_models'
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in trained_models.items():
            model_filename = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            model_path = os.path.join(models_dir, f'{model_filename}.joblib')
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'label_to_name': label_to_name,
            'feature_columns': feature_columns,
            'test_results': test_results,
            'y_test': y_test,
            'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(models_dir, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        # Store in session state
        st.session_state['trained_models'] = trained_models
        st.session_state['test_results'] = test_results
        st.session_state['y_test'] = y_test
        st.session_state['label_to_name'] = label_to_name
        st.session_state['models_trained'] = True
        
        st.success("✅ All models trained and saved successfully!")
        
        # Show summary
        st.subheader("Training Summary")
        summary_data = []
        for name in models.keys():
            summary_data.append({
                'Model': name,
                'Accuracy': f"{test_results[name]['accuracy']:.4f}",
                'Precision': f"{test_results[name]['precision']:.4f}",
                'Recall': f"{test_results[name]['recall']:.4f}",
                'F1-Score': f"{test_results[name]['f1_score']:.4f}",
                'Train Time (s)': f"{test_results[name]['train_time']:.2f}",
                'Test Time (s)': f"{test_results[name]['test_time']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False,
                                           key=lambda x: x.astype(float))
        st.dataframe(summary_df, use_container_width=True)

elif page == "View Performance":
    st.header("Model Performance Analysis")
    
    # Check if models are trained
    if 'models_trained' not in st.session_state or not st.session_state['models_trained']:
        st.warning("⚠️ Models not trained yet. Please go to 'Train Models' page first.")
        
        # Try to load saved models
        models_dir = 'saved_models'
        metadata_path = os.path.join(models_dir, 'metadata.joblib')
        
        if os.path.exists(metadata_path):
            st.info("📂 Found saved models. Loading...")
            metadata = joblib.load(metadata_path)
            st.session_state['test_results'] = metadata['test_results']
            st.session_state['label_to_name'] = metadata['label_to_name']
            st.session_state['y_test'] = metadata.get('y_test')
            st.session_state['models_trained'] = True
            st.success("✓ Loaded saved models!")
        else:
            st.stop()
    
    test_results = st.session_state.get('test_results', {})
    
    if not test_results:
        st.error("No performance data available.")
        st.stop()
    
    # Performance comparison plot
    st.subheader("Performance Comparison")
    required_metrics = {"accuracy", "precision", "recall", "f1_score", "train_time", "test_time"}
    is_full_metrics = all(required_metrics.issubset(test_results[name].keys()) for name in test_results.keys())
    if is_full_metrics:
        fig = plot_performance_comparison(test_results)
    else:
        fig = plot_performance_comparison_legacy(test_results)
    st.pyplot(fig)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    metrics_data = []
    if is_full_metrics:
        for name in test_results.keys():
            metrics_data.append({
                'Model': name,
                'Accuracy': test_results[name]['accuracy'],
                'Precision': test_results[name]['precision'],
                'Recall': test_results[name]['recall'],
                'F1-Score': test_results[name]['f1_score'],
                'Train Time (s)': test_results[name]['train_time'],
                'Test Time (s)': test_results[name]['test_time']
            })
    else:
        st.warning("⚠️ Performance data is from an older format. Retrain to show precision/recall and timing.")
        for name in test_results.keys():
            metrics_data.append({
                'Model': name,
                'Accuracy': test_results[name].get('accuracy'),
                'F1-Score': test_results[name].get('f1_score')
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    sort_column = 'Accuracy' if 'Accuracy' in metrics_df.columns else metrics_df.columns[1]
    metrics_df = metrics_df.sort_values(sort_column, ascending=False)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Best model
    best_model = metrics_df.iloc[0]['Model']
    if 'Accuracy' in metrics_df.columns:
        st.success(f"🏆 Best Model: **{best_model}** (Accuracy: {metrics_df.iloc[0]['Accuracy']:.4f})")
    else:
        st.success(f"🏆 Best Model: **{best_model}**")
    
    # Confusion matrix for selected model
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select Model", list(test_results.keys()))
    
    # Check if we have the necessary data to plot confusion matrix
    y_test = st.session_state.get('y_test')
    y_pred = test_results.get(selected_model, {}).get('y_pred') if selected_model in test_results else None
    label_to_name = st.session_state.get('label_to_name', {})
    
    if y_test is not None and y_pred is not None and label_to_name:
        fig_cm = plot_confusion_matrix(y_test, y_pred, label_to_name, selected_model)
        st.pyplot(fig_cm)
    else:
        st.info("ℹ️ Confusion matrix is only available for freshly trained models. Please train models first to see confusion matrices.")
        if y_test is None:
            st.warning("⚠️ Test labels (y_test) not available.")
        if y_pred is None:
            st.warning("⚠️ Predictions (y_pred) not available for this model.")

elif page == "Make Prediction":
    st.header("Make Predictions")
    
    # Check if models exist
    models_dir = 'saved_models'
    metadata_path = os.path.join(models_dir, 'metadata.joblib')
    
    if not os.path.exists(metadata_path):
        st.warning("⚠️ No trained models found. Please train models first.")
        st.stop()
    
    metadata = joblib.load(metadata_path)
    label_to_name = metadata['label_to_name']
    feature_columns = metadata['feature_columns']
    
    st.subheader("Enter Sensor Readings")
    st.info("Enter values for the following MQ sensor readings:")
    
    # Create input fields
    sensor_values = {}
    col1, col2 = st.columns(2)
    
    for i, col_name in enumerate(feature_columns):
        if i % 2 == 0:
            sensor_values[col_name] = col1.number_input(
                col_name, min_value=0, max_value=10000, value=100, key=f"input_{col_name}"
            )
        else:
            sensor_values[col_name] = col2.number_input(
                col_name, min_value=0, max_value=10000, value=100, key=f"input_{col_name}"
            )
    
    # Model selection (paper-aligned, with legacy fallback)
    model_names = list(metadata.get('test_results', {}).keys())
    model_file_map = {}

    if model_names:
        for name in model_names:
            model_filename = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            model_path = os.path.join(models_dir, f"{model_filename}.joblib")
            if os.path.exists(model_path):
                model_file_map[name] = model_path

    if not model_file_map:
        legacy_files = [f for f in glob.glob(os.path.join(models_dir, "*.joblib")) if "metadata" not in f]
        for path in legacy_files:
            label = map_model_filename_to_label(path)
            model_file_map[label] = path
        if any("Legacy" in name for name in model_file_map.keys()):
            st.info("ℹ️ Using legacy models (SVM/XGBoost). Retrain to align with the Hananto paper.")

    if not model_file_map:
        st.error("No model files found! Please train models first.")
        st.stop()

    selected_model_name = st.selectbox("Select Model", list(model_file_map.keys()))
    
    if st.button("🔮 Predict", type="primary"):
        # Load model
        model = joblib.load(model_file_map[selected_model_name])
        
        # Prepare input
        input_data = np.array([[sensor_values[col] for col in feature_columns]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
        
        # Display results
        predicted_class = label_to_name[prediction]
        
        st.success(f"🎯 **Predicted Fruit:** {predicted_class}")
        
        if probabilities is not None:
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Fruit': [label_to_name[i] for i in range(len(probabilities))],
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(prob_df, use_container_width=True)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(prob_df['Fruit'], prob_df['Probability'])
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_xlim([0, 1])
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Fruit Classification ML Pipeline | Built with Streamlit</p>", 
            unsafe_allow_html=True)
