# Autism Spectrum Disorder Classification

A machine learning classification project for predicting Autism Spectrum
Disorder (ASD) using demographic and screening-related features.

## Project Overview

This project follows an end-to-end machine learning workflow:

- Data cleaning
- Exploratory Data Analysis
- Feature preprocessing
- Handling class imbalance using SMOTENC
- Multiple machine learning models
- Hyperparameter optimization using RandomizedSearchCV
- Model evaluation using F1-score
- Final model training
- Prediction on unseen test data
- Streamlit interface for model inference

## Models

The project compares multiple classification algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Gradient Boosting
- XGBoost
- CatBoost

## Evaluation

F1-score is used as the primary metric because the target classes are
imbalanced.

## Streamlit Application

A lightweight Streamlit interface allows users to enter the required
features and obtain a prediction from the trained model.

