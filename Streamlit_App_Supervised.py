import io
from docx import Document
from docx.shared import Inches
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Title
st.title("IPS Fraud Detection Model Trainer & Evaluator")
st.write("Upload a training dataset and a testing dataset to evaluate supervised models.")

# Upload datasets
train_file = st.file_uploader("Upload Training Dataset (with isFraud)", type=["csv"])
test_file = st.file_uploader("Upload Testing Dataset (without isFraud column)", type=["csv"])
test_labels_file = st.file_uploader("Upload True Labels for Testing Dataset", type=["csv"])

# Function to check and preview uploaded files
def check_and_preview(file, name):
    if file is not None:
        file.seek(0)
        content = file.read()
        st.write(f"{name} file size: {len(content)} bytes")
        if len(content) == 0:
            st.error(f"{name} file is empty! Please upload a valid CSV.")
            return None
        file.seek(0)
        try:
            df = pd.read_csv(file)
            st.write(f"{name} dataset preview:")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"Error reading {name} file: {e}")
            return None
    return None

# Check and load dataframes
train_df = check_and_preview(train_file, "Training")
test_df = check_and_preview(test_file, "Testing")
test_labels_df = check_and_preview(test_labels_file, "Test Labels")

le = LabelEncoder()
le_fitted = False

def preprocess(df, fit_scaler=False, scaler=None, fit_label=False):
    global le, le_fitted
    df = df.copy()

    if 'transactionID' in df.columns:
        df = df.drop(columns=['transactionID'])

    if fit_label:
        le.fit(df['type'])
        le_fitted = True

    df['type'] = le.transform(df['type']) if le_fitted else df['type']

    X = df.drop(columns=['isFraud'], errors='ignore')

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler

if train_df is not None and test_df is not None and test_labels_df is not None:
    X_train, scaler = preprocess(train_df, fit_scaler=True, fit_label=True)
    y_train = train_df['isFraud']
    X_test, _ = preprocess(test_df, fit_scaler=False, scaler=scaler)
    y_test = test_labels_df['isFraud']

    # Check if "step" column is in test_df for detection time
    step_times = test_df["step"] * 0.1 if "step" in test_df.columns else None

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

    results = {}

    st.subheader("Model Evaluation Results")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        macro_avg = classification_rep['macro avg']
        weighted_avg = classification_rep['weighted avg']

        # Calculate average detection time for true positives
        if step_times is not None:
            tp_indices = (y_test == 1) & (y_pred == 1)
            avg_detection_time = step_times[tp_indices].mean()
        else:
            avg_detection_time = None

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "Macro Avg": macro_avg,
            "Weighted Avg": weighted_avg,
            "Confusion Matrix": cm,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "Avg Detection Time (s)": avg_detection_time
        }

        st.write(f"### {name}")
        st.write("**Classification Report**")
        st.text(classification_report(y_test, y_pred))

        if avg_detection_time is not None:
            st.write(f"**Average Detection Time (True Positives):** {avg_detection_time:.2f} seconds")

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f"{name} - Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f"{name} - ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

    # Summary Table
    st.subheader("Model Performance Summary")
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "AUC": metrics["AUC"],
            "Macro Avg F1": metrics["Macro Avg"]["f1-score"],
            "Weighted Avg F1": metrics["Weighted Avg"]["f1-score"],
            "Avg Detection Time (s)": metrics["Avg Detection Time (s)"]
        })
    summary = pd.DataFrame(summary_data)
    st.dataframe(summary)

    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="model_performance_summary.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload all three files: training, testing, and true test labels.")
