# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import glob
import joblib
import time
from datetime import datetime
import importlib.util

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score
)

# Optional XGBoost
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None

# -------------------------
# Page config + CSS (same style family)
# -------------------------
st.set_page_config(
    page_title="🧫 Fermentation Endpoint Detection | L. bulgaricus CFL1",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    font-size: 2.3rem;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-header p {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    opacity: 0.92;
}
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.25rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.10);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.14);
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
.small-muted {
    color: #666;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Paths (update if needed)
# -------------------------
DATA_FILE_MONITORING = "./Ferm_dataverse_files/Fermentation monitoring of L. delbrueckii subsp. bulgaricus CFL1.txt"
DATA_FILE_FUNCTIONAL  = "./Ferm_dataverse_files/Functional and physical properties of L. delbrueckii subsp. bulgaricus CFL1.txt"
DATA_FILE_MANIFEST    = "./Ferm_dataverse_files/MANIFEST.TXT"

MODELS_DIR = "saved_models_fermentation"
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# Helper: robust text reader
# -------------------------
def _read_txt_as_dataframe(path: str) -> pd.DataFrame:
    """
    Attempt to read a tab-delimited or semi-structured txt exported from Excel.
    We try separators in order: tab, semicolon, comma.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # First try to detect multi-row headers (prefix row + variable row)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if lines:
            def _split_row(line: str) -> list[str]:
                return [c.strip() for c in line.rstrip("\n").split("\t")]

            prefix_row_idx = None
            best_count = 0
            for i, line in enumerate(lines):
                cells = _split_row(line)
                count = sum(1 for c in cells if re.match(r"^[A-D]_\d+$", c))
                if count > best_count:
                    best_count = count
                    prefix_row_idx = i

            if prefix_row_idx is not None and best_count >= 5:
                prefix_row = _split_row(lines[prefix_row_idx])
                var_row_idx = None
                var_row = None
                for j in range(prefix_row_idx + 1, min(prefix_row_idx + 8, len(lines))):
                    cells = _split_row(lines[j])
                    low = " ".join(cells).lower()
                    if any(k in low for k in ["time", "ph", "od", "naoh", "gluc", "lactic", "dcw"]):
                        var_row_idx = j
                        var_row = cells
                        break

                if var_row_idx is not None and var_row is not None:
                    unit_row = _split_row(lines[var_row_idx + 1]) if var_row_idx + 1 < len(lines) else []
                    unit_text = " ".join(unit_row).lower()
                    data_start = var_row_idx + 2
                    if any(u in unit_text for u in ["(h)", "(c)", "g/l", "ml", "u_ph", "nm"]):
                        data_start = var_row_idx + 3

                    n_cols = max(len(prefix_row), len(var_row))
                    prefix_row += [""] * (n_cols - len(prefix_row))
                    var_row += [""] * (n_cols - len(var_row))

                    cols = []
                    seen = {}
                    for idx in range(n_cols):
                        pref = prefix_row[idx].strip()
                        var = var_row[idx].strip()
                        if pref and var:
                            name = f"{pref}_{var}"
                        elif pref:
                            name = pref
                        elif var:
                            name = var
                        else:
                            name = f"col_{idx}"

                        name = " ".join(name.replace(":", "").split())
                        if name in seen:
                            seen[name] += 1
                            name = f"{name}_{seen[name]}"
                        else:
                            seen[name] = 1
                        cols.append(name)

                    df = pd.read_csv(path, sep="\t", header=None, skiprows=data_start, engine="python")
                    if df.shape[1] > len(cols):
                        extra = [f"col_{i}" for i in range(len(cols), df.shape[1])]
                        cols = cols + extra
                    df.columns = cols[: df.shape[1]]
                    return df
    except Exception:
        # If anything goes wrong, fall back to simpler parsing below
        pass

    # Try tab
    for sep in ["\t", ";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            # Heuristic: require at least ~10 columns to consider it parsed properly
            if df.shape[1] >= 10:
                return df
        except Exception:
            continue

    # Fallback: read with python engine and infer
    df = pd.read_csv(path, sep=None, engine="python")
    return df

# -------------------------
# Wide -> Long extractor
# -------------------------
def discover_replicate_prefixes(columns) -> list:
    """
    Finds replicate prefixes like A_1, A_2, B_1, C_4, D_3 in the column names.
    According to the dataset documentation, columns are grouped by condition_replicate. (A_1, B_2, ...)
    """
    prefixes = set()
    for c in columns:
        if not isinstance(c, str):
            continue
        # common pattern: starts with "A_1", "B_3", etc.
        if len(c) >= 3 and c[0] in ["A", "B", "C", "D"] and c[1] == "_":
            # try to capture "A_1" or "A_10" (though doc shows 1..4)
            rest = c[2:]
            num = ""
            for ch in rest:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                prefixes.add(f"{c[0]}_{num}")
    return sorted(prefixes)

def find_signal_column_for_prefix(columns, prefix: str, signal_keywords: list) -> str | None:
    """
    For a given prefix like 'A_1', find the best-matching signal column
    among candidates containing both the prefix and at least one keyword.
    """
    candidates = []
    for c in columns:
        if not isinstance(c, str):
            continue
        if prefix in c:
            low = c.lower()
            if any(k in low for k in signal_keywords):
                candidates.append(c)
    if not candidates:
        return None

    # Prefer exact-ish matches
    preferred_order = [
        "time", "t", "od", "pH".lower(), "naoh", "gluc", "glucose", "lactic", "la"
    ]
    def score(col):
        low = col.lower()
        s = 0
        for i, tok in enumerate(preferred_order):
            if tok in low:
                s += (len(preferred_order) - i)
        # shorter names are often cleaner
        s += max(0, 20 - len(col))
        return s

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]

def wide_to_long_monitoring(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide monitoring dataframe into a long format:
      columns: prefix, condition, replicate, time_h, T_C, pH, OD880, NaOH_mL, gluc_gL, LA_gL, DCW_gL, OD600
    Some signals might be missing depending on how the export was done; we keep what we can find.
    """
    cols = list(df_wide.columns)
    prefixes = discover_replicate_prefixes(cols)
    if not prefixes:
        raise ValueError(
            "Could not detect replicate prefixes (A_1, B_2, ...). "
            "Check the header row / delimiter parsing."
        )

    # Define signal keyword sets (case-insensitive)
    keys_time = ["time", " t", "t "]
    keys_temp = ["temp", "temperature", " t("]
    keys_ph   = ["ph"]
    keys_od   = ["od", "880"]
    keys_naoh = ["naoh"]
    keys_gluc = ["gluc", "glucose"]
    keys_la   = [" la", "lactic", "lactate", "lactic acid"]
    keys_dcw  = ["dcw", "dry"]
    keys_od600= ["600"]

    records = []
    for pref in prefixes:
        c_time = find_signal_column_for_prefix(cols, pref, keys_time)
        c_temp = find_signal_column_for_prefix(cols, pref, keys_temp)
        c_ph   = find_signal_column_for_prefix(cols, pref, keys_ph)
        c_od   = find_signal_column_for_prefix(cols, pref, keys_od)
        c_naoh = find_signal_column_for_prefix(cols, pref, keys_naoh)
        c_gluc = find_signal_column_for_prefix(cols, pref, keys_gluc)
        c_la   = find_signal_column_for_prefix(cols, pref, keys_la)
        c_dcw  = find_signal_column_for_prefix(cols, pref, keys_dcw)
        c_od600= find_signal_column_for_prefix(cols, pref, keys_od600)

        # Build a small frame for this prefix using whatever columns exist
        sub = pd.DataFrame()
        for name, col in [
            ("time_h", c_time),
            ("T_C", c_temp),
            ("pH", c_ph),
            ("OD880", c_od),
            ("NaOH_mL", c_naoh),
            ("gluc_gL", c_gluc),
            ("LA_gL", c_la),
            ("DCW_gL", c_dcw),
            ("OD600", c_od600),
        ]:
            if col is not None and col in df_wide.columns:
                sub[name] = pd.to_numeric(df_wide[col], errors="coerce")

        # Must have time + at least one signal to be meaningful
        if "time_h" not in sub.columns or sub.drop(columns=["time_h"], errors="ignore").shape[1] == 0:
            continue

        sub["prefix"] = pref
        sub["condition"] = pref.split("_")[0]
        sub["replicate"] = int(pref.split("_")[1])

        # Drop rows where time is NaN
        sub = sub.dropna(subset=["time_h"]).copy()

        # Sort by time
        sub = sub.sort_values("time_h")

        # Deduplicate time if needed (keep first)
        sub = sub.drop_duplicates(subset=["time_h"], keep="first")

        records.append(sub)

    if not records:
        raise ValueError("No usable replicate blocks found after parsing wide columns.")
    long_df = pd.concat(records, ignore_index=True)

    # Enforce numeric time
    long_df["time_h"] = pd.to_numeric(long_df["time_h"], errors="coerce")
    long_df = long_df.dropna(subset=["time_h"]).copy()

    return long_df

# -------------------------
# Endpoint labeling (rule-based -> training target)
# -------------------------
def add_endpoint_labels(
    df_long: pd.DataFrame,
    signal: str,
    eps_mode: str = "percent_of_max_rate",
    eps_value: float = 5.0,
    consecutive: int = 2,
    min_time_h: float = 0.0
) -> pd.DataFrame:
    """
    Creates endpoint_reached labels using a rate threshold rule on the chosen signal.
    Endpoint is defined per (prefix) time series.

    - Compute rate = d(signal)/d(time)
    - Determine epsilon:
        * percent_of_max_rate: eps = (eps_value/100)*max(abs(rate))
        * absolute: eps = eps_value
    - endpoint_reached(t) = 1 if abs(rate) < eps for 'consecutive' consecutive steps, after min_time_h
    """
    if signal not in df_long.columns:
        raise ValueError(f"Signal '{signal}' not found. Available: {list(df_long.columns)}")

    out = []
    for pref, g in df_long.groupby("prefix"):
        g = g.sort_values("time_h").copy()
        y = g[signal].astype(float)

        dt = g["time_h"].diff()
        ds = y.diff()
        rate = ds / dt.replace(0, np.nan)

        # Replace inf/NaN
        rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        g["rate"] = rate

        # Epsilon
        if eps_mode == "percent_of_max_rate":
            max_rate = np.max(np.abs(rate.values)) if len(rate) else 0.0
            eps = (eps_value / 100.0) * max_rate
        else:
            eps = float(eps_value)

        g["eps"] = eps

        # Condition: low rate
        low = (np.abs(g["rate"].values) < eps).astype(int)

        # consecutive low-rate points => endpoint flag
        # rolling sum over last 'consecutive' points equals consecutive
        roll = pd.Series(low).rolling(consecutive, min_periods=consecutive).sum().fillna(0).values
        endpoint = (roll >= consecutive).astype(int)

        # Only valid after min_time_h
        endpoint = np.where(g["time_h"].values >= min_time_h, endpoint, 0)

        g["endpoint_reached"] = endpoint.astype(int)

        # Once reached, keep it reached (monotonic)
        g["endpoint_reached"] = np.maximum.accumulate(g["endpoint_reached"].values)

        out.append(g)

    return pd.concat(out, ignore_index=True)

def make_features(df_labeled: pd.DataFrame, signal: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Features for monitoring:
      - time
      - process variables (T, pH, NaOH) if present
      - signal value
      - rate
      - rolling stats (optional)
    Target:
      - endpoint_reached
    Group:
      - prefix (so we can avoid leakage across same fermentation run)
    """
    feature_cols = ["time_h"]
    for c in ["T_C", "pH", "NaOH_mL"]:
        if c in df_labeled.columns:
            feature_cols.append(c)

    # signal + rate
    feature_cols.append(signal)
    feature_cols.append("rate")

    X = df_labeled[feature_cols].copy()
    y = df_labeled["endpoint_reached"].astype(int).copy()
    groups = df_labeled["prefix"].copy()
    return X, y, groups

# -------------------------
# Models
# -------------------------
def define_models_binary():
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", "passthrough"),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample"
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11))
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ])
    }

    if HAS_XGBOOST:
        xgb = importlib.import_module("xgboost")
        XGBClassifier = getattr(xgb, "XGBClassifier")
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9
            ))
        ])

    return models

def group_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Split by fermentation run (prefix) to avoid leakage.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx], groups.iloc[test_idx]

def train_and_eval_binary(X_train, y_train, X_test, y_test, models):
    trained, results = {}, {}

    progress = st.progress(0)
    status = st.empty()
    total = len(models)

    for i, (name, model) in enumerate(models.items(), start=1):
        status.text(f"Training {name} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        status.text(f"Testing {name} ...")
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        test_time = time.perf_counter() - t1

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = None
        if len(np.unique(y_test)) == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_score)
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)

        trained[name] = model
        results[name] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "train_time": train_time,
            "test_time": test_time,
            "y_pred": y_pred
        }

        progress.progress(i / total)

    status.text("Done!")
    time.sleep(0.2)
    progress.empty()
    status.empty()
    return trained, results

def plot_metrics(results: dict):
    models = list(results.keys())
    metrics = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1_score")]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[0] for m in metrics])
    for idx, (title, key) in enumerate(metrics):
        r = idx // 2 + 1
        c = idx % 2 + 1
        fig.add_trace(go.Bar(x=models, y=[results[m][key] for m in models], name=title), row=r, col=c)

    fig.update_layout(height=650, showlegend=False)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(tickangle=25)
    return fig

def plot_confmat(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = px.imshow(
        cm,
        x=["No endpoint", "Endpoint"],
        y=["No endpoint", "Endpoint"],
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title=title
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return fig

# -------------------------
# Cached data loading
# -------------------------
@st.cache_data
def load_monitoring_long():
    df_wide = _read_txt_as_dataframe(DATA_FILE_MONITORING)
    df_long = wide_to_long_monitoring(df_wide)
    return df_long

st.markdown(
    """
    <div class='metric-card'>
        <b>Reference:</b> Tovilla-Coutino M.L. et al. (2022) — Effect of fermentation pH, temperature,
        and harvest on cell growth and functional properties of frozen and freeze-dried Lactobacillus delbrueckii subsp.
        bulgaricus CFL1, <a href='https://doi.org/10.15454/FZHIE0' target='_blank'>https://doi.org/10.15454/FZHIE0</a>,
        Recherche Data Gouv, V1
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📥 Load & Explore Data",
    "🏷️ Define Endpoint Labels",
    "🛠️ Train Endpoint Models",
    "📊 Model Comparison",
    "🔮 Online Endpoint Prediction"
])

# -------------------------
# HOME
# -------------------------
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🧫 Fermentation Endpoint Detection</h1>
        <p>Soft-sensor style monitoring for <b>L. delbrueckii subsp. bulgaricus</b> CFL1</p>
        <p>Fermentation Endpoint Monitoring Dashboard — Compare different labeling rules and ML models to detect when a fermentation process has effectively finished</p>
        <p style='margin: 0.35rem 0;'><small><strong>Problem statement:</strong> How can fermentation time-series signals (T, pH, OD, NaOH, and optional metabolites) be used to automatically detect the endpoint, improve consistency, and support smarter bioprocess monitoring?</small></p>
        <p style='margin: 0.35rem 0;'><small><strong>Why it matters:</strong> Helps operators make timely decisions, reduces variability and waste, and supports scalable, data‑driven fermentation control.</small></p>
        <p style='margin: 0.35rem 0;'><small><strong>Endpoint label:</strong> Effectively defined operationally from a rate-threshold rule on a chosen signal (e.g., OD880 plateau).</small></p>
        <p style='margin: 0.35rem 0;'><small><strong>Output:</strong> Trained ML models with accuracy, precision, recall, F1, and decision rules you can interpret for fermentation process control.</small></p>
        <p style='margin: 0.35rem 0;'><small><strong>Note:</strong> Use this app to explore how different labeling settings and ML models choices (e.g., Logistic Regression, Random Forest, KNN, SVM, XGBoost)
        affect endpoint detection performance.</small></p>

    </div>
    """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class="info-box">
    #     <b>Problem statement:</b> How can time‑series fermentation signals (T, pH, OD, NaOH, and optional metabolites)
    #     be used to automatically detect the endpoint, improve consistency, and support smarter bioprocess monitoring?
    # </div>
    # """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class="info-box">
    #     <b>Why it matters:</b> Helps operators make timely decisions, reduces variability and waste,
    #     and supports scalable, data‑driven fermentation control.
    # </div>
    # """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class="info-box">
    #     <b>Output:</b> Trained ML models for endpoint detection, clear performance charts,
    #     and decision rules you can interpret for quality control.
    # </div>
    # # """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class="info-box">
    #     <b>Note:</b> Use this app to explore how different labeling settings and model choices
    #     affect endpoint detection performance and interpretation.
    # </div>
    # """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Data source (loaded locally):</b><br/>
        • Fermentation monitoring time-series (T, pH, OD, NaOH; plus sampled gluc/LA/DCW when available)<br/>
        • Four conditions A–D (temperature × pH), multiple biological replicates, time 0 = inoculation<br/>
        <small><b>Condition</b> = a fixed setpoint pair (e.g., 42°C & pH 5.8). A–D are the four combinations.</small>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📘 Workflow, Metrics & Manifest Guide", expanded=False):
        st.markdown("""
        **Workflow**
        - Load the fermentation monitoring file and reshape it to long format.
        - Define endpoint labels with the rate‑threshold rule.
        - Split by fermentation run (group split) to avoid leakage.
        - Train multiple models and compare metrics on the test runs.

        **Metrics**
        - **Accuracy**: overall correctness.
        - **Precision**: when we predict endpoint, how often it is correct.
        - **Recall**: how many true endpoints we successfully detect.
        - **F1**: balance between precision and recall.

        **Manifest (what it is)**
        - The <b>MANIFEST.TXT</b> file is a simple file list (names + sizes).
        - Details about conditions A–D, replicate codes, and column layout are documented inside the data files themselves.
        - The main data files are:
          • <b>Fermentation monitoring</b>: time‑series signals used for endpoint detection.
          • <b>Functional & physical properties</b>: sampled outcomes at harvest times.
        - Use these files together to understand process dynamics and outcomes.
        """, unsafe_allow_html=True)

        if os.path.exists(DATA_FILE_MANIFEST):
            with open(DATA_FILE_MANIFEST, "rb") as f:
                st.download_button(
                    label="Open/Download MANIFEST.TXT",
                    data=f,
                    file_name="MANIFEST.TXT",
                    mime="text/plain"
                )
        if os.path.exists(DATA_FILE_MONITORING):
            with open(DATA_FILE_MONITORING, "rb") as f:
                st.download_button(
                    label="Open/Download Fermentation Monitoring (TXT)",
                    data=f,
                    file_name=os.path.basename(DATA_FILE_MONITORING),
                    mime="text/plain"
                )
        if os.path.exists(DATA_FILE_FUNCTIONAL):
            with open(DATA_FILE_FUNCTIONAL, "rb") as f:
                st.download_button(
                    label="Open/Download Functional & Physical Properties (TXT)",
                    data=f,
                    file_name=os.path.basename(DATA_FILE_FUNCTIONAL),
                    mime="text/plain"
                )

    st.markdown("""
    <div class="info-box">
    <h3>📋 About This Demo App</h3>
        <p>This demo app is a simple demonstration of how to build a data-driven classifier that detects when the fermentation endpoint is reached, using time, pH, NaOH, OD, and optional metabolites.</p>
        <b>How this app works:</b><br/>
        1) We <b>load and clean</b> the time-series measurements for each fermentation run.<br/>
        2) We <b>create an endpoint label</b> using a simple rule: when a signal stops changing for a while, we call it “endpoint”.<br/>
        3) We <b>train models</b> to learn that rule from data, then compare their performance.<br/>
        4) We <b>predict</b> for a new timepoint to answer: “Has the endpoint been reached?”
    </div>
    """, unsafe_allow_html=True)

    # Quick stats if possible
    try:
        df_long = load_monitoring_long()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (long)", f"{len(df_long):,}")
        c2.metric("Runs (prefix)", f"{df_long['prefix'].nunique()}")
        c3.metric("Conditions", f"{df_long['condition'].nunique()} (A–D)")
        c4.metric("Signals found", ", ".join([c for c in ["T_C","pH","OD880","NaOH_mL","gluc_gL","LA_gL","DCW_gL","OD600"] if c in df_long.columns]) or "—")
    except Exception as exc:
        st.markdown(f"<div class='warning-box'><b>Could not parse monitoring file.</b><br/><span class='small-muted'>{exc}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="metric-card">
        <b>Tip:</b> Use <b>Group split</b> (by fermentation run) to avoid leakage. The app does this by default.<br/>
        <small>What this means: all rows from the same run (e.g., A_1) go either to <b>train</b> or <b>test</b>, never both.
        This prevents the model from “seeing” parts of the same run in training and inflating results.</small>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# DATA EXPLORATION
# -------------------------
elif page == "📥 Load & Explore Data":
    st.header("Load & Explore Fermentation Monitoring Data")

    st.markdown("""
    <div class="info-box">
        <b>What you see here:</b><br/>
        • A preview table of the cleaned data (each row is one time point).<br/>
        • A simple chart so you can visually inspect how each signal changes over time.<br/>
        • Filters to focus on a specific condition or replicate.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading and reshaping monitoring data (wide → long)..."):
        df_long = load_monitoring_long()

    st.success(f"✓ Loaded long dataframe: {len(df_long):,} rows across {df_long['prefix'].nunique()} runs.")

    # Filters
    colA, colB = st.columns([1, 2])
    with colA:
        cond = st.multiselect("Condition (A–D = temperature × pH setpoints)", sorted(df_long["condition"].unique()), default=sorted(df_long["condition"].unique()))
        runs = st.multiselect("Runs (prefix)", sorted(df_long["prefix"].unique())[:8], default=sorted(df_long["prefix"].unique())[:2])
    with colB:
        signal_opts = [c for c in ["OD880", "pH", "NaOH_mL", "gluc_gL", "LA_gL", "DCW_gL", "OD600", "T_C"] if c in df_long.columns]
        signal = st.selectbox("Signal to plot", signal_opts, index=0 if signal_opts else 0)
        show_points = st.checkbox("Show points", value=False)

    df_view = df_long[df_long["condition"].isin(cond) & df_long["prefix"].isin(runs)].copy()
    st.dataframe(df_view.head(50), use_container_width=True)

    if signal_opts:
        fig = px.line(
            df_view.sort_values("time_h"),
            x="time_h",
            y=signal,
            color="prefix",
            markers=show_points,
            title=f"{signal} vs time (selected runs)"
        )
        fig.update_layout(xaxis_title="Time (h)", yaxis_title=signal)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No known signals found in parsed data. Check delimiter/header parsing.")

# -------------------------
# LABEL DEFINITION
# -------------------------
elif page == "🏷️ Define Endpoint Labels":
    st.header("Define Endpoint Labels (Operational Rule)")

    df_long = load_monitoring_long()

    st.markdown("""
    <div class="info-box">
        <b>Operational endpoint rule:</b><br/>
        We say “endpoint reached” when the chosen signal <b>stops changing</b> (or changes very slowly) for a short period.
        This is a practical rule used when a clear “end time” is not recorded.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Step-by-step labeling:</b><br/>
        1) Pick a signal (for example, OD880).<br/>
        2) Compute how fast it changes between time points (its “rate”).<br/>
        3) Mark points as “stable” when the rate is below ε.<br/>
        4) If we get N stable points in a row, we label “endpoint reached”.<br/>
        5) Once reached, it stays 1 for the rest of the run.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Parameter meanings:</b><br/>
        • <b>Signal</b>: the curve you want to watch (e.g., OD880, glucose, lactic acid).<br/>
        • <b>ε (epsilon)</b>: how “small” the change must be to count as “stable”.<br/>
        • <b>ε mode</b>:<br/>
        &nbsp;&nbsp;– <b>percent_of_max_rate</b>: ε is a percentage of the largest rate observed in the run.<br/>
        &nbsp;&nbsp;– <b>absolute</b>: ε is a fixed rate value you choose (same units as the signal per hour).<br/>
        • <b>N consecutive points</b>: how many stable points in a row before we declare endpoint.<br/>
        • <b>Ignore early time</b>: avoids labeling the very beginning as “endpoint”.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>How to interpret the preview plot:</b><br/>
        • The main line is the chosen signal over time.<br/>
        • The dotted line is the endpoint label (0 or 1).<br/>
        • A flat/slow‑changing signal should trigger the endpoint label after N stable points.
    </div>
    """, unsafe_allow_html=True)

    signal_opts = [c for c in ["OD880", "gluc_gL", "LA_gL", "pH", "NaOH_mL"] if c in df_long.columns]
    if not signal_opts:
        st.error("Could not find typical endpoint signals (OD880, gluc, LA, pH, NaOH) in parsed data.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        signal = st.selectbox("Signal for endpoint rule", signal_opts, index=0)
    with col2:
        eps_mode = st.selectbox("ε mode", ["percent_of_max_rate", "absolute"])
    with col3:
        eps_value = st.number_input("ε value", min_value=0.0001, value=5.0, step=0.5, help="Percent of max rate (if percent mode) or absolute rate threshold")
    with col4:
        consecutive = st.number_input("Consecutive points (N)", min_value=1, value=2, step=1)

    min_time_h = st.slider("Ignore early time before (h)", min_value=0.0, max_value=float(np.nanmax(df_long["time_h"])) if len(df_long) else 10.0, value=1.0, step=0.5)

    # Preview on a selected run
    run = st.selectbox("Preview on run (prefix)", sorted(df_long["prefix"].unique()))
    g = df_long[df_long["prefix"] == run].copy().sort_values("time_h")

    if st.button("🏷️ Generate labels", type="primary"):
        df_lab = add_endpoint_labels(
            df_long,
            signal=signal,
            eps_mode=eps_mode,
            eps_value=float(eps_value),
            consecutive=int(consecutive),
            min_time_h=float(min_time_h)
        )
        st.session_state["df_labeled"] = df_lab
        st.session_state["label_params"] = {
            "signal": signal, "eps_mode": eps_mode, "eps_value": float(eps_value),
            "consecutive": int(consecutive), "min_time_h": float(min_time_h)
        }
        st.success("✓ Labels created and stored in session.")

    df_lab = st.session_state.get("df_labeled", None)
    if df_lab is None:
        st.info("Create labels to continue.")
        st.stop()

    # Show preview plot for run
    g2 = df_lab[df_lab["prefix"] == run].copy().sort_values("time_h")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=g2["time_h"], y=g2[signal], mode="lines", name=signal), secondary_y=False)
    fig.add_trace(go.Scatter(x=g2["time_h"], y=g2["rate"], mode="lines", name="rate"), secondary_y=True)
    # Plot endpoint labels with distinct colors (0 = not reached, 1 = reached)
    not_reached = g2["endpoint_reached"] == 0
    reached = g2["endpoint_reached"] == 1
    fig.add_trace(go.Scatter(
        x=g2.loc[not_reached, "time_h"],
        y=g2.loc[not_reached, "endpoint_reached"],
        mode="markers",
        name="endpoint = 0 (not reached)",
        marker=dict(color="#ff6b6b", size=7, symbol="circle")
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=g2.loc[reached, "time_h"],
        y=g2.loc[reached, "endpoint_reached"],
        mode="markers",
        name="endpoint = 1 (reached)",
        marker=dict(color="#2ecc71", size=8, symbol="circle")
    ), secondary_y=True)

    # Vertical dashed line at first endpoint time
    if reached.any():
        first_endpoint_time = float(g2.loc[reached, "time_h"].iloc[0])
        fig.add_vline(
            x=first_endpoint_time,
            line_dash="dash",
            line_color="#2ecc71",
            annotation_text="endpoint start",
            annotation_position="top"
        )

    fig.update_layout(title=f"Endpoint labeling preview ({run})", height=450)
    fig.update_xaxes(title_text="Time (h)")
    fig.update_yaxes(title_text=signal, secondary_y=False)
    fig.update_yaxes(title_text="Rate / Endpoint", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Basic label stats
    pos_rate = float(g2["endpoint_reached"].mean())
    st.markdown(
        f"<div class='metric-card'><b>Run label prevalence:</b> endpoint_reached = 1 for {pos_rate*100:.1f}% of time points (after monotonic fill).</div>",
        unsafe_allow_html=True
    )

# -------------------------
# TRAIN MODELS
# -------------------------
elif page == "🛠️ Train Endpoint Models":
    st.header("Train Binary Endpoint Detection Models")

    df_lab = st.session_state.get("df_labeled", None)
    params = st.session_state.get("label_params", None)
    if df_lab is None or params is None:
        st.warning("⚠️ Please define endpoint labels first (page: Define Endpoint Labels).")
        st.stop()

    st.markdown("""
    <div class="info-box">
        <b>What happens during training:</b><br/>
        • The app turns each time point into simple features (time, pH, NaOH, signal, and rate).<br/>
        • We split by <b>fermentation run</b> so the model is tested on new runs it never saw.<br/>
        • We train multiple models and compare their accuracy, precision, recall, F1, and ROC-AUC.
    </div>
    """, unsafe_allow_html=True)

    signal = params["signal"]
    st.markdown(f"""
    <div class="metric-card">
        <b>Label definition:</b> signal={signal}, eps_mode={params['eps_mode']}, eps_value={params['eps_value']},
        consecutive={params['consecutive']}, min_time_h={params['min_time_h']}
    </div>
    """, unsafe_allow_html=True)

    X, y, groups = make_features(df_lab, signal=signal)

    # split settings
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.caption("Split is done by fermentation run (prefix) to reduce leakage.")

    if st.button("🔄 Train all models", type="primary"):
        X_train, X_test, y_train, y_test, g_train, g_test = group_split(
            X, y, groups, test_size=float(test_size), random_state=int(random_state)
        )

        st.info(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | Train runs: {g_train.nunique()} | Test runs: {g_test.nunique()}")
        train_counts = y_train.value_counts().to_dict()
        test_counts = y_test.value_counts().to_dict()
        st.caption(f"Train label counts: {train_counts} | Test label counts: {test_counts}")
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            st.warning("Only one class is present in the train/test split. Precision/Recall/F1 and ROC-AUC may be 0 or undefined.")
        models = define_models_binary()

        with st.spinner("Training..."):
            trained, results = train_and_eval_binary(X_train, y_train, X_test, y_test, models)

        # Save
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle = {
            "trained_models": trained,
            "results": results,
            "label_params": params,
            "feature_columns": list(X.columns),
            "signal": signal,
            "X_columns": list(X.columns),
            "y_test": y_test.values,
            "X_test": X_test,
            "groups_test": g_test,
            "saved_at": stamp
        }
        joblib.dump(bundle, os.path.join(MODELS_DIR, f"endpoint_models_{stamp}.joblib"))

        st.session_state["trained_bundle"] = bundle
        st.success("✅ Models trained & saved.")

        # summary table
        summary = []
        for name, r in results.items():
            summary.append({
                "Model": name,
                "Accuracy": r["accuracy"],
                "Balanced Acc": r["balanced_accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1_score"],
                "ROC-AUC": r["roc_auc"],
                "Train Time (s)": r["train_time"],
                "Test Time (s)": r["test_time"]
            })
        df_sum = pd.DataFrame(summary).sort_values("F1", ascending=False)
        st.dataframe(
            df_sum.style.format({
                "Accuracy": "{:.3f}",
                "Balanced Acc": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": (lambda v: "—" if pd.isna(v) else f"{v:.3f}"),
                "Train Time (s)": "{:.2f}",
                "Test Time (s)": "{:.2f}"
            }),
            use_container_width=True
        )

# -------------------------
# MODEL COMPARISON
# -------------------------
elif page == "📊 Model Comparison":
    st.header("Model Comparison & Confusion Matrix")

    bundle = st.session_state.get("trained_bundle", None)

    st.markdown("""
    <div class="info-box">
        <b>How to read the results:</b><br/>
        • <b>Accuracy</b>: overall correctness.<br/>
        • <b>Precision</b>: when the model says “endpoint”, how often it’s right.<br/>
        • <b>Recall</b>: how many true endpoints the model catches.<br/>
        • <b>F1</b>: balanced score that combines precision and recall.<br/>
        • <b>ROC-AUC</b>: ranking quality (higher is better; 0.5 is random).
    </div>
    """, unsafe_allow_html=True)

    # If not in session, try load latest saved
    if bundle is None:
        files = sorted(glob.glob(os.path.join(MODELS_DIR, "endpoint_models_*.joblib")))
        if not files:
            st.warning("⚠️ No trained models found yet. Train models first.")
            st.stop()
        latest = files[-1]
        bundle = joblib.load(latest)
        st.info(f"Loaded latest saved bundle: {os.path.basename(latest)}")

    results = bundle["results"]
    y_test = bundle["y_test"]

    st.plotly_chart(plot_metrics(results), use_container_width=True)

    # Pick model
    model_name = st.selectbox("Select model for confusion matrix", list(results.keys()))
    y_pred = results[model_name]["y_pred"]
    st.plotly_chart(plot_confmat(y_test, y_pred, f"Confusion Matrix — {model_name}"), use_container_width=True)

    # Metrics table (all models)
    df_metrics = pd.DataFrame([
        {
            "Model": k,
            "Accuracy": v["accuracy"],
            "Balanced Acc": v["balanced_accuracy"],
            "Precision": v["precision"],
            "Recall": v["recall"],
            "F1": v["f1_score"],
            "ROC-AUC": v["roc_auc"]
        }
        for k, v in results.items()
    ]).sort_values("F1", ascending=False)
    st.dataframe(
        df_metrics.style.format({
            "Accuracy": "{:.3f}",
            "Balanced Acc": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": (lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        }),
        use_container_width=True
    )

    # Best model banner
    df_rank = pd.DataFrame([
        {"Model": k, "F1": v["f1_score"], "Recall": v["recall"], "Precision": v["precision"], "Accuracy": v["accuracy"]}
        for k, v in results.items()
    ]).sort_values("F1", ascending=False)
    best = df_rank.iloc[0]
    st.markdown(
        f"<div class='success-box'><b>🏆 Best by F1:</b> {best['Model']} — F1={best['F1']:.3f}, Recall={best['Recall']:.3f}, Precision={best['Precision']:.3f}</div>",
        unsafe_allow_html=True
    )

# -------------------------
# ONLINE PREDICTION
# -------------------------
elif page == "🔮 Online Endpoint Prediction":
    st.header("Online Endpoint Prediction (Current Process State → Endpoint?)")

    files = sorted(glob.glob(os.path.join(MODELS_DIR, "endpoint_models_*.joblib")))
    if not files:
        st.warning("⚠️ No trained models found. Train models first.")
        st.stop()

    st.markdown("""
    <div class="info-box">
        <b>What this does:</b><br/>
        Enter the current process values (time, pH, NaOH, signal, rate). The model then predicts whether the endpoint
        has been reached <b>right now</b>. This is a simple “yes/no” decision with an optional probability.
    </div>
    """, unsafe_allow_html=True)

    selected_bundle_path = st.selectbox("Select a trained bundle", [os.path.basename(f) for f in files], index=len(files)-1)
    st.caption("A trained bundle is a saved package of models + label settings from a previous training run.")
    try:
        bundle = joblib.load(os.path.join(MODELS_DIR, selected_bundle_path))
    except ModuleNotFoundError as exc:
        st.error(
            "Could not load this bundle because a required library is missing in the current environment. "
            "This usually happens if the bundle was trained with a model (e.g., XGBoost) that is not installed here."
        )
        st.info(
            f"Missing module: {exc.name}. "
            "Fix options: install the missing package, or retrain models in this environment without that model."
        )
        st.stop()

    models = bundle["trained_models"]
    feature_cols = bundle["feature_columns"]
    signal = bundle["signal"]
    label_params = bundle["label_params"]

    st.markdown(f"""
    <div class="metric-card">
        <b>Bundle:</b> {selected_bundle_path}<br/>
        <b>Signal:</b> {signal}<br/>
        <b>Label rule:</b> {label_params}
    </div>
    """, unsafe_allow_html=True)

    # Input form for a single timepoint
    st.subheader("Enter current process values")

    # sensible defaults
    defaults = {c: 0.0 for c in feature_cols}
    defaults["time_h"] = 3.0
    if "pH" in defaults:
        defaults["pH"] = 5.2
    if "T_C" in defaults:
        defaults["T_C"] = 42.0
    if "NaOH_mL" in defaults:
        defaults["NaOH_mL"] = 10.0
    if signal in defaults:
        defaults[signal] = 1.0
    if "rate" in defaults:
        defaults["rate"] = 0.0

    # UI inputs
    cols = st.columns(2)
    inputs = {}
    for i, c in enumerate(feature_cols):
        target_col = cols[i % 2]
        inputs[c] = target_col.number_input(
            c,
            value=float(defaults.get(c, 0.0)),
            step=0.1
        )

    model_name = st.selectbox("Select model", list(models.keys()))
    if st.button("🔮 Predict endpoint", type="primary"):
        model = models[model_name]
        x = np.array([[inputs[c] for c in feature_cols]], dtype=float)

        pred = int(model.predict(x)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[0, 1])

        if pred == 1:
            st.success(f"✅ Endpoint reached (pred=1){' — P(endpoint)=%.3f' % prob if prob is not None else ''}")
        else:
            st.info(f"⏳ Not yet (pred=0){' — P(endpoint)=%.3f' % prob if prob is not None else ''}")

        # Probability bar
        if prob is not None:
            prob_df = pd.DataFrame({"Class": ["No endpoint", "Endpoint"], "Probability": [1 - prob, prob]})
            fig = px.bar(prob_df, x="Probability", y="Class", orientation="h", range_x=[0, 1], title="Prediction probability")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;padding:1rem;'>
  <p style='font-size:1.05rem;font-weight:600;'>🧫 Fermentation Endpoint Detection Demo</p>
  <p style='margin:0.35rem 0;'><small>
    Operational endpoint = rate threshold on a selected signal (per-run), then ML classifier learns to detect endpoint from current state.
  </small></p>
</div>
""", unsafe_allow_html=True)
