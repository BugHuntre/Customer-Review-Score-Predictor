# 📦 Customer Review Score Predictor

This project is an end-to-end machine learning pipeline built to predict **customer satisfaction scores** based on product and transaction features. It combines modular ML pipeline creation using **ZenML** and a user-friendly local app built with **Streamlit**.

---

## 🚀 Project Highlights

- ✅ **End-to-End ML Workflow** with ZenML
- 🧠 **LightGBM** model (with options for RandomForest & XGBoost)
- 🛠️ Modular design for easy experimentation and reproducibility
- 🌐 **Streamlit App** for local model interaction and prediction
- 💾 Model saving and loading for persistent results
- 📊 Performance comparison and feature importance visualization

---

## 🔍 Problem Statement

E-commerce businesses need insights into what drives customer satisfaction. This project predicts customer review scores using features such as:

- Payment details
- Product specs
- Freight costs
- Product descriptions and dimensions

---

## 🧩 Tech Stack

- **Python**
- **ZenML** – for reproducible pipelines
- **LightGBM** – primary ML model
- **Streamlit** – interactive UI for predictions
- **Scikit-learn**, **Pandas**, **Joblib**, etc.

---

## 🧪 Project Structure

```bash
.
├── data/                            # Raw dataset
├── steps/                           # ZenML pipeline steps (ingest, clean, train, evaluate)
├── pipeline/                        # Training pipeline definition
├── src/model_dev/                   # Custom model classes
├── saved_model/                     # Trained model binaries
├── run_pipeline.py                 # Entry point to train model
├── app.py                          # Streamlit app
├── _assets/                         # Pipeline images and visualizations
└── README.md                        # You're here

