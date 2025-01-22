# INF6027_INTRO_TO_DATA_SCIENCE
# Overview
This project explore stock classification using financial indicators and machine learning models.The dataset consists of financial data from publicly traded US companies between 2014 and 2018,providing 200+ financial indicators per stock.The primary objective is to classify stocks as "Buy" or "Not Buy" and to predict classes using logistic regression and XGBOOST.
# Project Structure
data/: Contains raw and processed datasets.
notebooks/: R Notebooks for data analysis and modelling.
README.md: Project overview and instructions.
# Research Questions
1.  How effectively can logistic regression classify stock movements ("buy" or "not to buy") based on engineered financial features?
2.	How do feature selection methods, such as information value and multicollinearity handling, improve the model's predictive accuracy?
3.	Which financial indicators exhibit the strongest predictive power for stock movements, and how are these indicators distributed across different sectors?
4.	How does the predictive performance of linear models Logistic Regression compare with non-linear models XGBoost for forecasting stock price variation and classification?
# Methodology
# Data Preprocessing
Handling missing values using median imputation.

Addressing outliers using IQR method.

Merging datasets from 2016,2017,2018 into one dataframe.
# Feature Selection
Information value-IV.Evaluate the predictive power of each feature
Handling multicollinearity
# Exploratory Data Analysis
Exploring the distribution of numeric features
Exploring the relationship between target variable (class) and key predictors
# Modelling
Evaluating Logistic Regression and XGBOOST model
Models evaluation by accuracy.
# Results
Logistic regression accuracy: 56.87
XGBOOST accuracy:67.75


