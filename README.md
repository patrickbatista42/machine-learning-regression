# Machine Learning Regression Project

This project implements a regression system using machine learning for car price prediction. Below is the project structure and description of each component.

## Project Structure

### Main Files
- `train_mod.csv` - Original dataset containing car data
- `README.md` - This documentation file

### PreProcessamento/ Folder
Contains scripts related to data preprocessing:
- `main.py` - Main script that executes the entire preprocessing pipeline
- `preprocessamento.py` - Helper functions for data preprocessing
- Generated files:
  - `ProcessedDatabase_SEM_outliers.csv` - Processed dataset without outliers
  - `ProcessedDatabase_target_SEM_outliers.csv` - Target variable (prices) without outliers

### Estudo_metodos/ Folder
Contains scripts for model implementation and evaluation:
- `mainTest.py` - Main script for model testing
- `modelos.py` - Implementation of different machine learning models
- `otimizacao.py` - Functions for hyperparameter optimization
- `feature_engineering.py` - Functions for feature engineering
- `avaliacao.py` - Metrics and model evaluation functions

### Processed Files
Different versions of the processed dataset:
- `ProcessedDatabase_SEM_outliers.csv` - Complete dataset without outliers
- `ProcessedDatabase_target_SEM_outliers.csv` - Only prices (target) without outliers
- `ProcessedDatabase_target.csv` - Base version of prices

## Preprocessing Pipeline

Data preprocessing includes:
1. Missing data cleaning
2. Outlier treatment
3. Categorical variable encoding
4. Data normalization
5. Correlation analysis
6. Dimensionality reduction (PCA)

## Implemented Models

The machine learning models include:
1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Regression (SVR)

## How to Use

1. Run preprocessing:
```bash
cd PreProcessamento
python main.py
```

2. Run models:
```bash
cd Estudo_metodos
python mainTest.py
```

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Results
Processing results are saved in the corresponding CSV files, allowing for subsequent analysis and comparison between different approaches (with/without outliers).
