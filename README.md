# Random Forest Model for Classification

## Overview
This project implements a **Random Forest Classifier** to predict the target variable using a dataset of features. The model is optimized using **Optuna for hyperparameter tuning**, and various exploratory data analysis (EDA) techniques are used to understand the dataset. The project also integrates **SHAP** and **Partial Dependence Plots (PDP)** for model interpretability.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [License](#license)

## Project Structure

project/ │ 
         ├── data/ │
                   ├── X_Train_Data_Input.csv │
                   ├── Y_Train_Data_Target.csv │
                   ├── X_Test_Data_Input.csv │
                   ├── Y_Test_Data_Target.csv │
         ├── src/ │
                  ├── rf_model.py # Main script for training and evaluating the model │
                  ├── utils.py # Utility functions for data preprocessing, visualization, etc. │
         ├── models/ │
                     ├── final_rf_model.pkl # Saved Random Forest model │
         ├── analysis/ │ 
                       ├── exploratory data analysis.ipynb # Jupyter notebook for exploratory data analysis │
         ├── README.md 
         └── requirements.txt


## Features
- **Random Forest Classifier**: A robust model for classification tasks.
- **Optuna Hyperparameter Tuning**: Automated hyperparameter tuning for optimal performance.
- **Exploratory Data Analysis (EDA)**: Includes visualizations and correlation analysis.
- **Model Interpretability**: SHAP values and Partial Dependence Plots (PDP) for understanding feature importance and effects.
- **Pipeline Implementation**: Streamlined preprocessing and model fitting.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Create and activate a virtual environment:
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
3. Install the required packages:
    pip install -r requirements.txt

## Usage
Run the model training script:

python scripts/rf_model.ipynb
This script will preprocess the data, train the model, evaluate its performance, and save the trained model.

## Exploratory Data Analysis (EDA):

Open analysis/exploratory data analysis.ipynb in Jupyter Notebook to explore the data.
Evaluate and Interpret the Model:

The script outputs model performance metrics such as accuracy, confusion matrix, ROC-AUC score, and SHAP values.

Exploratory Data Analysis (EDA)
The EDA notebook includes:

** Visualizations like histograms, bar charts, and box plots.
Correlation matrix and heatmaps to explore relationships between features.
** Outlier analysis and handling of missing values.
## Model Training and Evaluation
The model is trained using a Random Forest Classifier.
Data is split into training, validation, and test sets.
Performance metrics such as accuracy, cross-validation scores, confusion matrix, and ROC-AUC score are calculated.
## Hyperparameter Tuning with Optuna
Optuna is used to optimize the hyperparameters of the Random Forest model:

n_estimators: Number of trees in the forest.
max_depth: Maximum depth of each tree.
min_samples_split: Minimum number of samples required to split a node.
min_samples_leaf: Minimum number of samples required at a leaf node.
Model Interpretability
SHAP values: Used to visualize feature importance and understand their effects on predictions.
Partial Dependence Plots (PDP): Illustrates the relationship between features and predicted outcomes.

## Saving and Loading the Model
The trained model is saved using joblib:

joblib.dump(final_rf_model, 'models/final_rf_model.pkl')

**To load the model:**

model = joblib.load('models/final_rf_model.pkl')

## License
This project is licensed under the MIT License.
