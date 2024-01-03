# Home Credit Default Risk - Classification

## Overview
This project revolves around constructing a classification model aimed at predicting the likelihood of loan default among applicants. The repository encompasses several notebooks covering Exploratory Data Analysis (EDA), Data Preparation, Cleaning, and the handling of additional datasets. Additionally, it includes a notebook dedicated to the test file, a package, and a pipeline for streamlined processing.

# Table of Contents
* [Introduction](#introduction)
* [General Info](#general-info)
* [Dependencies](#dependencies)
* [Project Structure](#project-structure)
* [Utilization](#utilization)

## Introduction

Ensuring fair opportunities for all applicants to access loans is crucial. However, financial institutions also need to accurately assess an applicant's ability to repay to safeguard their business interests. This project aims to develop a classification model that predicts the likelihood of loan default risk for applicants.

## General Info

In this project, I began by balancing the target class using the random under-sampling method and then dove into thorough data visualization for a comprehensive analysis. Next, I focused on removing outliers and performed data imputation using both statistical methods and the KNeighbors algorithm to predict missing values based on correlated features.

Following data cleaning, I delved into feature engineering to enhance the predictive capabilities of the model. Scaling numerical features and encoding categorical ones were vital steps in this process. Subsequently, I trained four classification algorithms: logistic regression, random forest, XGBoost, and artificial neural networks. Notably, the artificial neural network demonstrated superior performance among these classifiers.

Throughout this project, I gained valuable insights into leveraging statistical methods, efficient data-cleaning techniques, and maximizing the power of data visualization for effective analysis.

## Dependencies
This project is created with:
- Python version: 3.11.3
- Pandas version: 2.0.1
- Numpy version: 1.24.3
- Matplotlib package version: 3.7.1
- Seaborn package version: 0.12.2
- imbalanced-learn version: 0.11.0
- category-encoders version: 2.6.1
- Keras version: 2.14.0
- XGBoost version: 1.7.6

## Project Structure
- **Data**: Access the dataset [here]([https://www.kaggle.com/competitions/widsdatathon2019/data](https://www.kaggle.com/competitions/home-credit-default-risk/overview)).
- **home_default_EDA**: This Jupyter Notebook encompasses exploratory data analysis, data cleaning, and extensive data visualization to understand the dataset.
- **home_default_Prep**: This Jupyter Notebook involves feature engineering, scaling, encoding categorical variables, and training various models for predictive analysis.
- **home_default_Test**: This notebook is dedicated to working on the test file and making predictions.
- **home_default_exData**: Within this notebook, extra datasets are prepared and integrated with the training dataset to enrich the overall dataset.
- **home_default_package**: This Python file comprises essential functions for streamlined use within the project's pipeline.
- **home_default_Pipeline**: The pipeline developed here is designed for training and deploying a classification model focused on home default risk assessment.

- # Utilization
To utilize this project, please download the dataset from the above link. Subsequently, download the 'home_default_package.py' and 'home_default_Pipeline' files and execute them using the downloaded file paths.
