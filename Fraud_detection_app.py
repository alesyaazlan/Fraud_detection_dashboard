import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Base Path
BASE_PATH = 'C:/Users/wesya/.vscode/FYP'

#Page Config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

THEME = {
    "bg":           "#F0F7F4",
    "sidebar":      "#FFFFFF",
    "primary":      "#FF6B4A",   
    "teal":         "#4ECDC4",   
    "card":         "#FFFFFF",
    "text":         "#2D3142",
    "muted":        "#8A94A6",
    # per-model palette for CV chart
    "lr":           "#FF6B4A",
    "dt":           "#F9C74F",
    "rf":           "#4ECDC4",
    "svm":          "#A78BFA",
    "xgb":          "#3B82F6",
    # multi-metric palette
    "metrics": ["#FF6B4A", "#4ECDC4", "#F9C74F", "#A78BFA"],
}

# INJECT CSS
st.markdown(f"""
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] {{
    background-color: {THEME['bg']};
}}
[data-testid="stHeader"] {{
    background-color: {THEME['bg']};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {THEME['sidebar']};
    border-right: 1px solid #E8EDF2;
}}
[data-testid="stSidebar"] * {{
    color: {THEME['text']} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background-color: {THEME['card']};
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border: 1px solid #EEF2F7;
}}
[data-testid="stMetricValue"] {{
    color: {THEME['primary']} !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {THEME['muted']} !important;
    font-size: 13px !important;
}}

/* ── Headings & text ── */
h1, h2, h3 {{
    color: {THEME['text']} !important;
    font-weight: 700 !important;
}}
p, div, span, label {{
    color: {THEME['text']};
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {THEME['primary']};
    color: white;
    border: none;
    border-radius: 8px;
}}
.stButton > button:hover {{
    background-color: #e5563a;
    color: white;
}}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div {{
    background-color: {THEME['card']};
    border-radius: 8px;
    border: 1px solid #EEF2F7;
}}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {{
    background-color: {THEME['card']};
    border-radius: 12px;
    border: 1px solid #EEF2F7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}

/* ── Divider ── */
hr {{
    border-color: #E8EDF2;
}}

/* ── Info / warning boxes ── */
.stAlert {{
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# MATPLOTLIB GLOBAL STYLE
plt.rcParams.update({
    "figure.facecolor":  THEME["card"],
    "axes.facecolor":    THEME["card"],
    "axes.edgecolor":    "#E8EDF2",
    "axes.labelcolor":   THEME["text"],
    "axes.titlecolor":   THEME["text"],
    "xtick.color":       THEME["muted"],
    "ytick.color":       THEME["muted"],
    "text.color":        THEME["text"],
    "grid.color":        "#EEF2F7",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#E8EDF2",
    "font.family":       "sans-serif",
})

# HELPER: save-fig with transparent outer background
def styled_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["card"])
    return fig, ax

def styled_figs(rows, cols, figsize=(12, 4)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(THEME["card"])
    return fig, axes


# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('Fraud_detection/Data/Raw/Credit_Card_Fraud_Prediction_by_Kelvin_Kelu_555K.csv')
    return df

@st.cache_data
def preprocess_data():
    df = load_data()
    df = df.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last',
                           'street', 'trans_num', 'unix_time'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], dayfirst=True)
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df = df.drop(columns=['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'], dayfirst=True)
    df['age'] = 2020 - df['dob'].dt.year
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    df = df.drop(columns=['dob'])
    cat_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job', 'zip']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

@st.cache_data
def get_predictions():
    df = preprocess_data()
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = joblib.load(f'{BASE_PATH}/Fraud_detection/Models/scaler.pkl')
    imputer = joblib.load(f'{BASE_PATH}/Fraud_detection/Models/imputer.pkl')
    X_test = scaler.transform(X_test)
    X_test = imputer.transform(X_test)
    return X_test, y_test

# LOAD MODELS 
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': {
            'smote': joblib.load('Fraud_detection/Models/lr_smote.pkl'),
            'csl': joblib.load('Fraud_detection/Models/lr_csl.pkl')
        },
        'Decision Tree': {
            'smote': joblib.load('Fraud_detection/Models/dt_smote.pkl'),
            'csl': joblib.load('Fraud_detection/Models/dt_csl.pkl')
        },
        'Random Forest': {
            'smote': joblib.load('Fraud_detection/Models/rf_smote.pkl'),
            'csl': joblib.load('Fraud_detection/Models/rf_csl.pkl')
        },
        'SVM': {
            'smote': joblib.load(f'{BASE_PATH}/Fraud_detection/Models/svm_smote.pkl'),
            'csl': joblib.load(f'{BASE_PATH}/Fraud_detection/Models/svm_csl.pkl')
        },
        'XGBoost': {
            'smote': joblib.load(f'{BASE_PATH}/Fraud_detection/Models/xgb_smote.pkl'),
            'csl': joblib.load(f'{BASE_PATH}/Fraud_detection/Models/xgb_csl.pkl')
        }
    }
    return models

# SIDEBAR NAVIGATION 
st.sidebar.title(" Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    " Home",
    " Transaction Analysis",
    " Model Comparison",
    " Individual Model Results"
])

# LOAD EVERYTHING
df_raw = load_data()
X_test, y_test = get_predictions()
models = load_models()

#  HOME PAGE
if page == " Home":
    st.title(" Credit Card Fraud Detection")
    st.markdown("### Using Machine Learning to Detect Fraudulent Transactions")
    st.markdown("---")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df_raw):,}")
    with col2:
        st.metric("Fraud Cases", f"{df_raw['is_fraud'].sum():,}")
    with col3:
        st.metric("Fraud Rate", f"{df_raw['is_fraud'].mean()*100:.2f}%")
    with col4:
        st.metric("Legitimate Cases", f"{(df_raw['is_fraud']==0).sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Fraud vs Non-Fraud pie chart
    with col1:
        st.subheader(" Fraud vs Non-Fraud Distribution")
        fig, ax = styled_fig()
        labels = ['Non-Fraud', 'Fraud']
        sizes = [df_raw['is_fraud'].value_counts()[0], df_raw['is_fraud'].value_counts()[1]]
        colors = [THEME['teal'], THEME['primary']]
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Average Transaction Amount bar chart
    with col2:
        st.subheader(" Average Transaction Amount")
        fig, ax = styled_fig()
        avg_amt = df_raw.groupby('is_fraud')['amt'].mean()
        bars = ax.bar(['Non-Fraud', 'Fraud'], avg_amt.values,
                      color=[THEME['teal'], THEME['primary']], width=0.5,
                      edgecolor='white', linewidth=0.8)
        ax.set_ylabel('Average Amount ($)')
        ax.bar_label(bars, fmt='$%.2f', color=THEME['text'])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader(" Project Overview")
    st.markdown("""
    This dashboard presents a fraud detection system built using machine learning.
    
    **Models Used:**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    -  eXtreme Gradient Boosting (XGBOOST) 
    
    **Techniques to Handle Imbalanced Data:**
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Cost-Sensitive Learning
    
    **Dataset:** Credit Card Fraud Prediction by Kelvin Kelu (555,719 transactions)
    """)

#  TRANSACTION ANALYSIS PAGE
elif page == " Transaction Analysis":
    st.title(" Transaction Analysis")
    st.markdown("---")

    # Fraud by Category
    st.subheader(" Fraud by Category")
    fraud_cat = df_raw[df_raw['is_fraud']==1]['category'].value_counts()
    fig, ax = styled_fig(figsize=(10, 4))
    bars = ax.bar(fraud_cat.index, fraud_cat.values,
                  color='#3B82F6', edgecolor='white', linewidth=0.8)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Fraud Cases')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(bars)
    st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Fraud by Gender
    with col1:
        st.subheader(" Fraud by Gender")
        fraud_gender = df_raw[df_raw['is_fraud']==1]['gender'].value_counts()
        fig, ax = styled_fig()
        ax.pie(fraud_gender.values, labels=fraud_gender.index,
               autopct='%1.1f%%', colors=[THEME['teal'], THEME['primary']])
        ax.axis('equal')
        st.pyplot(fig)

    # Fraud by Hour
    with col2:
        st.subheader(" Fraud by Hour of Day")
        df_raw['trans_date_trans_time'] = pd.to_datetime(
            df_raw['trans_date_trans_time'], dayfirst=True)
        df_raw['hour'] = df_raw['trans_date_trans_time'].dt.hour
        fraud_hour = df_raw[df_raw['is_fraud']==1]['hour'].value_counts().sort_index()
        fig, ax = styled_fig()
        ax.plot(fraud_hour.index, fraud_hour.values,
                color=THEME['primary'], marker='o', linewidth=2,
                markerfacecolor=THEME['card'], markeredgewidth=2)
        ax.fill_between(fraud_hour.index, fraud_hour.values,
                        alpha=0.15, color=THEME['primary'])
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Fraud Cases')
        ax.set_xticks(range(0, 24))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        st.pyplot(fig)

    st.markdown("---")

    # Amount Distribution
    st.subheader("Transaction Amount Distribution")
    fig, axes = styled_figs(1, 2, figsize=(12, 4))

    for ax_i, (scale, title) in enumerate(zip([None, 'log'], ['Normal Scale', 'Log Scale'])):
        axes[ax_i].hist(df_raw[df_raw['is_fraud']==0]['amt'], bins=50,
                color=THEME['teal'], label='Non-Fraud', edgecolor='white')
        axes[ax_i].hist(df_raw[df_raw['is_fraud']==1]['amt'], bins=50,
                color=THEME['primary'], label='Fraud', zorder=3)
        axes[ax_i].set_xlabel('Transaction Amount ($)')
        axes[ax_i].set_ylabel('Frequency' if scale is None else 'Frequency (Log Scale)')
        axes[ax_i].set_title(title)
        axes[ax_i].legend()
        if scale:
            axes[ax_i].set_yscale(scale)

    st.pyplot(fig)

#  MODEL COMPARISON PAGE
elif page == " Model Comparison":
    st.title(" Model Comparison")
    st.markdown("---")

    # Compute metrics for all models
    results = []
    for model_name, model_versions in models.items():
        for method, model in model_versions.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, y_pred, output_dict=True)
            results.append({
                'Model': model_name,
                'Method': 'SMOTE' if method == 'smote' else 'Cost-Sensitive',
                'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
                'Precision': round(report['1']['precision'], 4),
                'Recall': round(report['1']['recall'], 4),
                'F1-Score': round(report['1']['f1-score'], 4),
                'Accuracy': round(report['accuracy'], 4)
            })

    results_df = pd.DataFrame(results)

    # Results Table
    st.subheader(" Model Performance Summary")
    st.dataframe(results_df, use_container_width=True)
    st.markdown("---")

    # Colour helper: SMOTE = teal, Cost-Sensitive = coral
    def method_colors(col): 
        return [THEME['teal'] if m=='SMOTE' else THEME['primary'] for m in col]

    x = np.arange(len(results_df))
    xlabels = [f"{r['Model']}\n({r['Method']})" for _, r in results_df.iterrows()]
    legend_handles = [
        plt.Rectangle((0,0),1,1, color=THEME['teal'],    label='SMOTE'),
        plt.Rectangle((0,0),1,1, color=THEME['primary'], label='Cost-Sensitive')
    ]

    # ROC-AUC Comparison
    st.subheader(" ROC-AUC Comparison")
    fig, ax = styled_fig(figsize=(10, 4))
    bars = ax.bar(x, results_df['ROC-AUC'], color=method_colors(results_df['Method']),
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('ROC-AUC Score'); ax.set_ylim(0.8, 1.0)
    ax.bar_label(bars, fmt='%.4f'); ax.legend(handles=legend_handles)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)
    st.markdown("---")

    # F1-Score Comparison
    st.subheader(" F1-Score Comparison (Fraud Class)")
    fig, ax = styled_fig(figsize=(10, 4))
    bars = ax.bar(x, results_df['F1-Score'], color=method_colors(results_df['Method']),
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('F1-Score')
    ax.bar_label(bars, fmt='%.4f'); ax.legend(handles=legend_handles)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)
    st.markdown("---")

    # Precision Comparison
    st.subheader("Precision Comparison (Fraud Class)")
    fig, ax = styled_fig(figsize=(10, 4))
    bars = ax.bar(x, results_df['Precision'], color=method_colors(results_df['Method']),
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('Precision')
    ax.bar_label(bars, fmt='%.4f'); ax.legend(handles=legend_handles)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)
    st.markdown("---")

    # Recall Comparison
    st.subheader("Recall Comparison (Fraud Class)")
    fig, ax = styled_fig(figsize=(10, 4))
    bars = ax.bar(x, results_df['Recall'], color=method_colors(results_df['Method']),
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('Recall')
    ax.bar_label(bars, fmt='%.4f'); ax.legend(handles=legend_handles)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)
    st.markdown("---")

    # Combined Grouped Bar Chart
    st.subheader("Combined Metrics Comparison")
    fig, ax = styled_fig(figsize=(14, 6))
    metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    n_models = len(results_df)
    n_metrics = len(metrics)
    bar_width = 0.15
    for i, metric in enumerate(metrics):
        positions = np.arange(n_models) + i * bar_width
        ax.bar(positions, results_df[metric], width=bar_width,
               label=metric, color=THEME['metrics'][i], edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(n_models) + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.set_title('All Metrics Comparison Across Models')
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)
    st.markdown("---")

    # Cross Validation Results
    st.subheader("Cross Validation Results")
    import json
    with open(f'{BASE_PATH}/Fraud_detection/Models/cv_results.json', 'r') as f:
        cv_results_loaded = json.load(f)

    cv_data = {
        'Model': list(cv_results_loaded.keys()),
        'Fold 1': [v['fold_scores'][0] for v in cv_results_loaded.values()],
        'Fold 2': [v['fold_scores'][1] for v in cv_results_loaded.values()],
        'Fold 3': [v['fold_scores'][2] for v in cv_results_loaded.values()],
        'Fold 4': [v['fold_scores'][3] for v in cv_results_loaded.values()],
        'Fold 5': [v['fold_scores'][4] for v in cv_results_loaded.values()],
        'Mean ROC-AUC': [round(v['mean'], 4) for v in cv_results_loaded.values()],
        'Standard Deviation': [round(v['std'], 4) for v in cv_results_loaded.values()]
    }
    cv_df = pd.DataFrame(cv_data)
    st.dataframe(cv_df, use_container_width=True)
    st.markdown("---")

    # CV Chart
    st.subheader("Cross Validation Mean ROC-AUC")
    fig, ax = styled_fig(figsize=(12, 5))
    cv_colors = [
        THEME['lr'],  THEME['lr'],   # LR
        THEME['dt'],  THEME['dt'],   # DT
        THEME['rf'],  THEME['rf'],   # RF
        THEME['svm'], THEME['svm'],  # SVM
        THEME['xgb'], THEME['xgb']  # XGBoost
    ]
    bars = ax.bar(cv_df['Model'], cv_df['Mean ROC-AUC'],
                  color=cv_colors, yerr=cv_df['Standard Deviation'], capsize=5,
                  edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Mean ROC-AUC'); ax.set_ylim(0.75, 1.05)
    ax.set_xticklabels(cv_df['Model'], rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.4f', padding=5)
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color=THEME['lr'],  label='Logistic Regression'),
        plt.Rectangle((0,0),1,1, color=THEME['dt'],  label='Decision Tree'),
        plt.Rectangle((0,0),1,1, color=THEME['rf'],  label='Random Forest'),
        plt.Rectangle((0,0),1,1, color=THEME['svm'], label='SVM'),
        plt.Rectangle((0,0),1,1, color=THEME['xgb'], label='XGBoost')
    ])
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig)

#  INDIVIDUAL MODEL RESULTS PAGE
elif page == " Individual Model Results":
    st.title(" Individual Model Results")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select Model",
                               ['Logistic Regression', 'Decision Tree',
                                'Random Forest', 'SVM', 'XGBoost'])
    with col2:
        selected_method = st.selectbox("Select Method", ['SMOTE', 'Cost-Sensitive'])

    method_key = 'smote' if selected_method == 'SMOTE' else 'csl'
    model = models[selected_model][method_key]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    report = classification_report(y_test, y_pred, output_dict=True)
    with col1:
        st.metric("ROC-AUC",   f"{roc_auc_score(y_test, y_prob):.4f}")
    with col2:
        st.metric("Precision", f"{report['1']['precision']:.4f}")
    with col3:
        st.metric("Recall",    f"{report['1']['recall']:.4f}")
    with col4:
        st.metric("F1-Score",  f"{report['1']['f1-score']:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader(" Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = styled_fig()
        # Custom colormap using the theme teal → primary
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "theme", ["#FFFFFF", THEME['teal'], THEME['primary']], N=256)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'], ax=ax,
                    linewidths=1, linecolor=THEME['bg'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        st.pyplot(fig)

    # ROC Curve
    with col2:
        st.subheader(" ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = styled_fig()
        ax.plot(fpr, tpr, color=THEME['primary'], linewidth=2.5,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.10, color=THEME['primary'])
        ax.plot([0, 1], [0, 1], color=THEME['muted'], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        st.pyplot(fig)

    st.markdown("---")

    # Classification Report
    st.subheader(" Classification Report")
    report_df = pd.DataFrame(classification_report(
        y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    if selected_model in ['Random Forest', 'XGBoost', 'Decision Tree']:
        st.subheader("Feature Importance")
        feature_names = ['merchant', 'category', 'amt', 'gender', 'city',
                        'state', 'zip', 'lat', 'long', 'city_pop', 'job',
                        'merch_lat', 'merch_long', 'hour', 'day',
                        'month', 'day_of_week', 'age']
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig, ax = styled_fig(figsize=(10, 6))
        
        bars = ax.barh(feature_imp['Feature'], feature_imp['Importance'],
               color=THEME['teal'], edgecolor='white', linewidth=0.6)

        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance — {selected_model} ({selected_method})')
        ax.invert_yaxis()
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        st.pyplot(fig)
    
    else:
        st.info("Feature Importance is only available for Decision Tree, Random Forest and XGBoost models.")