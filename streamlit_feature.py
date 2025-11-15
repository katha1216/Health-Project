import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(layout="wide", page_title="Healthcare Feature Engineering")

# --- Helper Functions (from your notebook) ---
# We use st.cache_data to speed up the app by caching results

@st.cache_data
def generate_synthetic_data():
    """Generates the synthetic dataset from the notebook."""
    st.info("No file uploaded. Using synthetic demo data.")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "patient_id": np.arange(1, n+1),
        "age": np.random.randint(18, 90, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "height_cm": np.random.normal(165, 10, n).round(1),
        "weight_kg": np.random.normal(70, 15, n).round(1),
        "systolic_bp": np.random.randint(100, 180, n),
        "diastolic_bp": np.random.randint(60, 110, n),
        "fasting_glucose_mg_dl": np.random.normal(100, 25, n).round(1),
        "hba1c_pct": np.random.normal(5.8, 1.0, n).round(2),
        "hdl_mg_dl": np.random.normal(50, 10, n).round(1),
        "smoker": np.random.choice([0, 1], n),
        "num_hospitalizations": np.random.randint(0, 6, n),
        "num_visits_last_year": np.random.randint(1, 12, n),
        "last_visit_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
        "medications": np.random.choice(["Metformin", "Statin", "Antihypertensive", "None"], n),
        "doctor_notes": np.random.choice([
            "Follow-up needed", "Stable condition", "Monitor BP", 
            "Elevated sugar levels", "Requires diet change"], n),
        "readmitted_30days": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    return df

@st.cache_data
def clean_data(df):
    """Applies data cleaning steps."""
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()
    df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    df_cleaned["last_visit_date"] = pd.to_datetime(df_cleaned["last_visit_date"])
    return df_cleaned

@st.cache_data
def engineer_features(df):
    """Applies feature engineering steps."""
    df_engineered = df.copy()
    
    # Handle potential division by zero if height is 0
    df_engineered["height_m_sq"] = (df_engineered["height_cm"] / 100) ** 2
    df_engineered["bmi"] = df_engineered.apply(
        lambda row: (row["weight_kg"] / row["height_m_sq"]) if row["height_m_sq"] > 0 else np.nan,
        axis=1
    ).round(2)
    
    df_engineered["bp_mean"] = (df_engineered["systolic_bp"] + 2 * df_engineered["diastolic_bp"]) / 3
    
    # Use a fixed date for consistent 'days_since_last_visit' calculation, as in the notebook
    reference_date = pd.Timestamp("2024-12-31")
    df_engineered["days_since_last_visit"] = (reference_date - df_engineered["last_visit_date"]).dt.days
    
    df_engineered["is_hypertensive"] = ((df_engineered["systolic_bp"] > 140) | (df_engineered["diastolic_bp"] > 90)).astype(int)
    
    # Handle potential NaNs in doctor_notes before .str.len()
    df_engineered["note_len"] = df_engineered["doctor_notes"].fillna("").str.len()
    
    # Log transforms
    df_engineered["log_hba1c_pct"] = np.log1p(df_engineered["hba1c_pct"])
    df_engineered["log_fasting_glucose_mg_dl"] = np.log1p(df_engineered["fasting_glucose_mg_dl"])
    df_engineered["log_days_since_last_visit"] = np.log1p(df_engineered["days_since_last_visit"])
    df_engineered["log_num_hospitalizations"] = np.log1p(df_engineered["num_hospitalizations"])
    
    # Drop intermediate columns
    df_engineered = df_engineered.drop(columns=["height_m_sq"])
    
    return df_engineered

@st.cache_data
def get_model_results(df_engineered):
    """Preprocesses data, trains RF, and returns results."""
    df_model = df_engineered.copy()
    
    # Define features and target
    X = df_model.drop(columns=["patient_id", "readmitted_30days", "doctor_notes", "last_visit_date"])
    y = df_model["readmitted_30days"]

    # Separate numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    
    # Ensure all columns are either num or cat (excluding bools if any)
    # For this app, let's cast 'smoker' and 'is_hypertensive' to object to be safe with OHE
    X['smoker'] = X['smoker'].astype(str)
    X['is_hypertensive'] = X['is_hypertensive'].astype(str)
    
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Build preprocessing pipeline (as in notebook)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ])

    # Create the full classifier pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Get feature importances
    model = clf.named_steps["model"]
    encoder = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = encoder.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(encoded_cat_features)
    
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return accuracy, report, cm, feat_imp

# --- Streamlit App UI ---
st.title("🏥 Healthcare Feature Engineering Pipeline")

# Sidebar for file upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your patient data (CSV)", type="csv")
demo_button = st.sidebar.button("Load Demo Data")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif demo_button:
    df = generate_synthetic_data()

if df is not None:
    st.header("1. Raw Data")
    st.dataframe(df.head())
    
    # --- Pipeline Execution ---
    st.header("2. Pipeline Execution")
    
    with st.spinner("Cleaning data..."):
        df_cleaned = clean_data(df)
    
    with st.spinner("Engineering new features..."):
        df_engineered = engineer_features(df_cleaned)
    st.success("Data cleaning and feature engineering complete!")
    
    # --- Results Tabs ---
    tab1, tab2, tab3 = st.tabs(["Engineered Data", "Visualizations", "Model Performance"])
    
    with tab1:
        st.subheader("Engineered Dataset")
        st.dataframe(df_engineered.head())
        
        # Download button
        csv = df_engineered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Engineered CSV",
            data=csv,
            file_name="engineered_healthcare_data.csv",
            mime="text/csv",
        )
        st.subheader("Data Description")
        st.write(df_engineered.describe())

    with tab2:
        st.subheader("Data Visualizations (on Engineered Data)")
        
        # Set style for plots
        sns.set(style="whitegrid", palette="muted")
        
        # Plot 1: Target variable distribution
        st.write("#### Target Distribution: Readmitted within 30 days")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(x="readmitted_30days", data=df_engineered, palette="Set2", ax=ax1)
        st.pyplot(fig1)

        # Plots in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot 2: Age distribution
            st.write("#### Age Distribution of Patients")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            sns.histplot(df_engineered["age"], bins=20, kde=True, color="teal", ax=ax2)
            st.pyplot(fig2)

        with col2:
            # Plot 3: BMI vs Systolic BP
            st.write("#### BMI vs Systolic Blood Pressure")
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.scatterplot(data=df_engineered, x="bmi", y="systolic_bp", hue="is_hypertensive", alpha=0.6, ax=ax3)
            plt.xlabel("BMI")
            plt.ylabel("Systolic BP (mmHg)")
            st.pyplot(fig3)
        
        # Plot 4: Correlation heatmap
        st.write("#### Correlation Heatmap (Numeric Features)")
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        # Select only numeric columns for correlation
        numeric_df = df_engineered.select_dtypes(include=np.number)
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax4)
        st.pyplot(fig4)

    with tab3:
        st.subheader("Predictive Model Performance")
        
        with st.spinner("Training Random Forest model on engineered features..."):
            try:
                accuracy, report, cm, feat_imp = get_model_results(df_engineered)
                
                st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
                
                st.subheader("Classification Report")
                st.text(report)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Confusion Matrix
                    st.write("#### Confusion Matrix")
                    fig5, ax5 = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax5)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(fig5)

                with col4:
                    # Feature Importances
                    st.write("#### Top 20 Feature Importances")
                    fig6, ax6 = plt.subplots(figsize=(8, 10))
                    top_20_features = feat_imp.nlargest(20)
                    sns.barplot(x=top_20_features.values, y=top_20_features.index, palette="viridis", ax=ax6)
                    plt.xlabel("Importance")
                    plt.ylabel("Feature")
                    st.pyplot(fig6)

            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
                st.error("This can happen if the uploaded data is very small or has a different structure than expected (e.g., missing 'readmitted_30days' column).")

else:
    st.info("Upload a CSV file or click 'Load Demo Data' in the sidebar to start the pipeline.")
    st.subheader("About this App")
    st.write("""
    This app replicates the feature engineering pipeline from your Jupyter Notebook.
    It performs the following steps:
    1.  **Loads Data**: You can upload your own CSV or use the synthetic demo data.
    2.  **Cleans Data**: Handles missing values and duplicates.
    3.  **Engineers Features**: Creates `bmi`, `bp_mean`, `days_since_last_visit`, `is_hypertensive`, `note_len`, and several log-transformed features.
    4.  **Visualizes Data**: Shows key distributions and correlations from the new data.
    5.  **Trains Model**: Builds a Random Forest classifier on the engineered features.
    6.  **Shows Results**: Displays model accuracy, a classification report, a confusion matrix, and the most important features.
    7.  **Export**: Allows you to download the final engineered dataset.
    """)