# 📱 Phone Price Prediction

This project builds a machine learning model to accurately predict mobile phone prices based on their technical specifications such as camera quality, internal memory, battery, CPU cores, and more.

## 🔍 Overview

We used data preprocessing techniques to clean and prepare the dataset, handled missing and outlier values, and engineered new features to improve model performance. Multiple regression models were trained and evaluated, including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

The best-performing model achieved:
- ✅ **92% Accuracy** on the training set
- ✅ **87% Accuracy** on the test set

## 📊 Features Used

- Internal Memory
- Rear and Front Camera (MP)
- CPU Cores
- Weight
- Battery
- Sale Price (discount)
- More...

## 🚀 Deployment

We plan to deploy the final model using:
- `Flask` or `Streamlit` for the web interface
- `pickle` or `joblib` for saving the trained model
- Hosting options: GitHub Pages / Render / HuggingFace Spaces

## 📁 Files

- `Phone Price Prediction.ipynb` — Full notebook with data cleaning, training & evaluation
- `model.pkl` — Saved trained model (for deployment)
- `README.md` — This file

## 📦 Installation

```bash
git clone https://github.com/Youssef-kotb/Phone-Price-Prediction.git
cd Phone-Price-Prediction
pip install -r requirements.txt
