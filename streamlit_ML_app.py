import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

# Set page config
st.set_page_config(
    page_title="Fruit Classification FROM MQ SENSOR DATA ML Pipeline",
    page_icon="🍎 MQ SENSOR DATA",
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
    """Define all models with preprocessing pipelines"""
    models = {
        'ANN (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            ))
        ])
    }
    return models

def train_and_evaluate(X_temp, y_temp, X_test, y_test, models):
    """Train models and evaluate"""
    # K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Cross-validation
    total_models = len(models)
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Performing cross-validation for {name}...")
        cv_scores = cross_val_score(model, X_temp, y_temp, cv=kfold, scoring='accuracy', n_jobs=-1)
        cv_f1_scores = cross_val_score(model, X_temp, y_temp, cv=kfold, scoring='f1_macro', n_jobs=-1)
        
        cv_results[name] = {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'f1_mean': cv_f1_scores.mean(),
            'f1_std': cv_f1_scores.std()
        }
        progress_bar.progress((idx + 1) / total_models / 2)  # Half progress for CV
    
    # Train final models
    trained_models = {}
    test_results = {}
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        model.fit(X_temp, y_temp)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        test_results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'y_pred': y_pred
        }
        progress_bar.progress(0.5 + (idx + 1) / total_models / 2)
    
    status_text.text("Training complete!")
    progress_bar.empty()
    status_text.empty()
    
    return trained_models, cv_results, test_results

def plot_performance_comparison(cv_results, test_results):
    """Plot performance comparison (accuracy & F1-score, with error bars for CV) -- classic bar plot version."""
    import seaborn as sns
    import pandas as pd
    import numpy as np

    models = list(cv_results.keys())

    # Prepare table for plotting
    acc_cv = [cv_results[m]['accuracy_mean'] for m in models]
    acc_cv_std = [cv_results[m]['accuracy_std'] for m in models]
    acc_test = [test_results[m]['accuracy'] for m in models]

    f1_cv = [cv_results[m]['f1_mean'] for m in models]
    f1_cv_std = [cv_results[m]['f1_std'] for m in models]
    f1_test = [test_results[m]['f1_score'] for m in models]

    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.set(style="whitegrid", font_scale=1.15)

    # Accuracy
    width = 0.35
    axes[0].bar(x - width/2, acc_cv, width, yerr=acc_cv_std, capsize=6, label="CV Accuracy", color="tab:blue")
    axes[0].bar(x + width/2, acc_test, width, label="Test Accuracy", color="tab:orange")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha='right')
    axes[0].set_ylim(0.7, 1.01)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Models")
    axes[0].legend(title="", loc="lower right")

    # F1-score
    axes[1].bar(x - width/2, f1_cv, width, yerr=f1_cv_std, capsize=6, label="CV F1-score", color="tab:blue")
    axes[1].bar(x + width/2, f1_test, width, label="Test F1-score", color="tab:orange")
    axes[1].set_title("Model F1-Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha='right')
    axes[1].set_ylim(0.7, 1.01)
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Models")
    axes[1].legend(title="", loc="lower right")

    plt.tight_layout()
    return fig

    # --- Plot F1-score ---
    for metric, mkr, color in zip(["CV F1-score", "Test F1-score"], ["o", "s"], ["tab:blue", "tab:orange"]):
        df_plot = df_f1[df_f1["Metric"] == metric]
        axes[1].scatter(
            np.arange(len(models)), 
            df_plot["Value"], 
            label=metric, 
            marker=mkr, 
            s=90, 
            c=color
        )
        # Only apply error bars for CV metrics
        if metric == "CV F1-score":
            axes[1].errorbar(
                np.arange(len(models)),
                df_plot["Value"],
                yerr=df_plot["Std"],
                fmt='none',
                ecolor=color,
                elinewidth=2,
                capsize=7
            )

    axes[1].set_title("Model F1-Score")
    axes[1].set_xticks(np.arange(len(models)))
    axes[1].set_xticklabels(models, rotation=30, ha='right')
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("Models")
    axes[1].set_ylim(0.7, 1.01)
    axes[1].legend(title="", loc="lower right")

    plt.tight_layout()
    return fig

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
st.markdown('<h1 class="main-header">🍎 Fruit Classification ML Pipeline</h1>', unsafe_allow_html=True)

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
    st.markdown("**Models to be trained:**")
    model_names = [
        "K-Nearest Neighbors",
        "Support Vector Machine (SVM)",
        "Logistic Regression",
        "Artificial Neural Network (MLPClassifier)",
        "XGBoost Classifier"
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
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.info(f"📊 Dataset split: Train+Val ({len(X_temp)} samples), Test ({len(X_test)} samples)")
        
        # Define models
        models = define_models()
        
        # Train and evaluate
        with st.spinner("Training models... This may take a few minutes."):
            trained_models, cv_results, test_results = train_and_evaluate(
                X_temp, y_temp, X_test, y_test, models
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
            'test_results': {k: {'accuracy': v['accuracy'], 'f1_score': v['f1_score']} 
                           for k, v in test_results.items()},
            'cv_results': cv_results,
            'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(models_dir, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        # Store in session state
        st.session_state['trained_models'] = trained_models
        st.session_state['cv_results'] = cv_results
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
                'CV Accuracy': f"{cv_results[name]['accuracy_mean']:.4f} ± {cv_results[name]['accuracy_std']:.4f}",
                'Test Accuracy': f"{test_results[name]['accuracy']:.4f}",
                'Test F1-Score': f"{test_results[name]['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test Accuracy', ascending=False, 
                                           key=lambda x: x.str.split().str[0].astype(float))
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
            st.session_state['cv_results'] = metadata['cv_results']
            st.session_state['test_results'] = metadata['test_results']
            st.session_state['label_to_name'] = metadata['label_to_name']
            st.session_state['models_trained'] = True
            st.success("✓ Loaded saved models!")
        else:
            st.stop()
    
    cv_results = st.session_state.get('cv_results', {})
    test_results = st.session_state.get('test_results', {})
    
    if not cv_results or not test_results:
        st.error("No performance data available.")
        st.stop()
    
    # Performance comparison plot
    st.subheader("Performance Comparison")
    fig = plot_performance_comparison(cv_results, test_results)
    st.pyplot(fig)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    metrics_data = []
    for name in cv_results.keys():
        metrics_data.append({
            'Model': name,
            'CV Accuracy (Mean)': f"{cv_results[name]['accuracy_mean']:.4f}",
            'CV Accuracy (Std)': f"{cv_results[name]['accuracy_std']:.4f}",
            'Test Accuracy': f"{test_results[name]['accuracy']:.4f}",
            'CV F1-Score (Mean)': f"{cv_results[name]['f1_mean']:.4f}",
            'CV F1-Score (Std)': f"{cv_results[name]['f1_std']:.4f}",
            'Test F1-Score': f"{test_results[name]['f1_score']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Test Accuracy', ascending=False,
                                       key=lambda x: x.str.split().str[0].astype(float))
    st.dataframe(metrics_df, use_container_width=True)
    
    # Best model
    best_model = metrics_df.iloc[0]['Model']
    st.success(f"🏆 Best Model: **{best_model}** (Test Accuracy: {metrics_df.iloc[0]['Test Accuracy']})")
    
    # Confusion matrix for selected model
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select Model", list(cv_results.keys()))
    
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
    
    # Model selection
    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    model_files = [f for f in model_files if 'metadata' not in f]
    
    if not model_files:
        st.error("No model files found!")
        st.stop()
    
    selected_model_file = st.selectbox("Select Model", 
                                      [os.path.basename(f).replace('.joblib', '') for f in model_files])
    
    if st.button("🔮 Predict", type="primary"):
        # Load model
        model_path = os.path.join(models_dir, f"{selected_model_file}.joblib")
        model = joblib.load(model_path)
        
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
