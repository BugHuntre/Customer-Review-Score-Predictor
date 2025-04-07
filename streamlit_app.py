import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import os
import subprocess
import matplotlib.pyplot as plt

PREDICTION_HISTORY_FILE = "prediction_history.csv"

def main():
    st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")
    st.title("📦 End-to-End Customer Satisfaction Pipeline with ZenML")

    model_name = "LightGBM"

    # Train pipeline
    if st.sidebar.button("⚙️ Train Pipeline"):
        st.sidebar.info("Training pipeline started with LightGBM...")
        try:
            subprocess.run(["python", "run_pipeline.py", model_name], check=True)
            st.sidebar.success("✅ Training pipeline completed!")
        except subprocess.CalledProcessError:
            st.sidebar.error("❌ Pipeline failed. Check terminal for details.")

    # Load model
    MODEL_PATH = f"saved_model/{model_name}.pkl"
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if not model:
        st.warning(f"Model file not found at {MODEL_PATH}. Please train the model.")

    # Pipeline images
    st.image(Image.open("_assets/high_level_overview.png"), caption="High Level Pipeline")

    st.markdown("""### Problem Statement  
    Predict customer satisfaction score based on features like order status, payment details, product specs, etc.  
    This pipeline is built using [ZenML](https://zenml.io/).""")

    st.image(Image.open("_assets/training_and_deployment_pipeline_updated.png"), caption="ZenML Pipeline Architecture")

    st.markdown("""### Feature Descriptions  
    | Feature | Description |
    |--------|-------------|
    | Payment Sequential | Order of payments |
    | Payment Installments | Installments chosen |
    | Payment Value | Total paid |
    | Price | Product price |
    | Freight Value | Delivery cost |
    | Product Name Length | Characters in name |
    | Product Description Length | Characters in description |
    | Product Photos Quantity | Number of photos |
    | Product Weight (g) | Weight in grams |
    | Product Dimensions (cm) | Length, Height, Width |
    """)

    # User Inputs
    st.sidebar.header("🔧 Input Features")
    inputs = {
        "payment_sequential": st.sidebar.slider("Payment Sequential", 0, 10, 1),
        "payment_installments": st.sidebar.slider("Payment Installments", 1, 24, 1),
        "payment_value": st.sidebar.number_input("Payment Value", 0.0),
        "price": st.sidebar.number_input("Price", 0.0),
        "freight_value": st.sidebar.number_input("Freight Value", 0.0),
        "product_name_lenght": st.sidebar.number_input("Product Name Length", 0),
        "product_description_lenght": st.sidebar.number_input("Product Description Length", 0),
        "product_photos_qty": st.sidebar.number_input("Product Photos Quantity", 0),
        "product_weight_g": st.sidebar.number_input("Product Weight (g)", 0.0),
        "product_length_cm": st.sidebar.number_input("Product Length (cm)", 0.0),
        "product_height_cm": st.sidebar.number_input("Product Height (cm)", 0.0),
        "product_width_cm": st.sidebar.number_input("Product Width (cm)", 0.0),
    }

    if st.button("🔍 Predict"):
        if model:
            df = pd.DataFrame([inputs])
            prediction = model.predict(df)
            pred_score = round(prediction[0], 2)
            st.success(f"🎯 Predicted Customer Satisfaction Score: **{pred_score}**")

            # Save prediction to history CSV
            df["prediction"] = pred_score
            if os.path.exists(PREDICTION_HISTORY_FILE):
                df.to_csv(PREDICTION_HISTORY_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(PREDICTION_HISTORY_FILE, index=False)

            # Show download button
            st.download_button(
                label="⬇️ Download Prediction",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_result.csv",
                mime="text/csv"
            )
        else:
            st.error("No trained model available for prediction.")

    if st.button("📊 Results Overview"):
        st.write("Model Performance:")
        results_df = pd.DataFrame({
            "Metrics": ["MSE", "RMSE"],
            "LightGBM": [1.804, 1.343]
        }).set_index("Metrics")
        st.dataframe(results_df)

        st.write("📈 Feature Importance (Dynamic from LightGBM):")
        fig, ax = plt.subplots()
        feature_names = model.feature_name_
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        ax.set_title("Feature Importance (Gain)")
        st.pyplot(fig)

    if os.path.exists(PREDICTION_HISTORY_FILE):
        st.write("📜 Past Predictions:")
        hist_df = pd.read_csv(PREDICTION_HISTORY_FILE)
        st.dataframe(hist_df.tail(5))  # Show latest 5 predictions

if __name__ == "__main__":
    main()
