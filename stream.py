import streamlit as st
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

st.title("XGBoost and LightGBM Regression Trainer with Visualizations")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.write(df.head())

    # Select target variable
    target_column = st.selectbox("Select target column", df.columns)

    # Convert categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting data
    test_size = st.slider("Test set size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    model_choice = st.selectbox("Select Model", ["XGBoost Regressor", "LightGBM Regressor"])

    if st.button("Train Model"):
        if model_choice == "XGBoost Regressor":
            model = xgb.XGBRegressor(objective='reg:squarederror')
        else:
            model = lgb.LGBMRegressor()

        st.write("Training the model, please wait...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.histplot(y_test - y_pred, bins=20, kde=True, ax=ax)
        ax.set_title("Error Distribution (Actual - Predicted)")
        st.pyplot(fig)

        if model_choice in ["XGBoost Regressor", "LightGBM Regressor"]:
            feature_importance = model.feature_importances_
            feature_names = X.columns

            fig, ax = plt.subplots()
            sns.barplot(x=feature_importance, y=feature_names, ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)
