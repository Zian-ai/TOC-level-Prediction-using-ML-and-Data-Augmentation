# Decision Trees (DT), Random Forest (RF), Support Vector Regression (SVR), Gradient Boost (GB), XGBoost (XGB), LassoLars (LL), Ridge (RR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import optuna
import os

# NSE Function
def nse(y_true, y_pred):
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (np.sum((y_true - y_pred) ** 2) / denominator)

# R2 Calculation Function
def calculate_r2(observed, predicted):
    observed_mean = np.mean(observed)
    predicted_mean = np.mean(predicted)
    numerator = np.sum((observed - observed_mean) * (predicted - predicted_mean))
    denominator = np.sqrt(np.sum((observed - observed_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2))
    
    if denominator == 0:
        return np.nan
    return (numerator / denominator) ** 2


# # File Paths   SACHEON & DOCHEON (REAL DATA)
# file_paths = [
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_DATA_TRAIN_MONTHLY.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_DATA_TRAIN_MONTHLY.csv"
# ]


# File Paths   SACHEON (WGAN DATA)
file_paths = [
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-100-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-200-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-300-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-500-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1000-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1500-Epochs.csv",
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-3000-Epochs.csv"
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(180-Synthetic-data).csv"
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(300-Synthetic-data).csv"
    r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(600-Synthetic-data).csv"
]

# # File Paths   DOCHEON (WGAN DATA)
# file_paths = [
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-100-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-200-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-300-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1000-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-3000-Epochs.csv"
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(180-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(300-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(600-Synthetic-data).csv"
# ]


# Features and Target
features = ['prcp', 'tmax', 'tmin', 'rhum']
target = 'TOC'
date_column = 'month' 

# Models Dictionary with corresponding Optuna parameter ranges
models = {
    'Decision Tree': {
        'model': DecisionTreeRegressor,
        'params': lambda trial: {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
    },
    'Gradient Boost': {
        'model': GradientBoostingRegressor,
        'params': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
        }
    },
    'XGBoost': {
        'model': XGBRegressor,
        'params': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor,
        'params': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
    },
    'Support Vector Regression': {
        'model': SVR,
        'params': lambda trial: {
            'C': trial.suggest_float('C', 0.1, 1000, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1, log=True),
        }
    },
    'Ridge': {
        'model': Ridge,
        'params': lambda trial: {
            'alpha': trial.suggest_float('alpha', 0.01, 1000, log=True),
        }
    },
    'Lasso': {
        'model': Lasso,
        'params': lambda trial: {
            'alpha': trial.suggest_float('alpha', 0.01, 1000, log=True),
        }
    }
}

# Results List
results = []

# Loop through each file and apply models with Optuna optimization
for file_path in file_paths:
    try:
        # Load Data
        data = pd.read_csv(file_path)
        X = data[features]
        y = data[target]
        dates = data[date_column]

        # Impute missing values in X
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=features)

        # Drop rows where the target variable is NaN
        y = y.dropna()
        X = X.loc[y.index]
        dates = dates.loc[y.index]

        # Split data into 80% training and 20% testing
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.05, random_state=42, shuffle=False
        )

        # Loop through each model
        for model_name, model_info in models.items():
            def objective(trial):
                params = model_info['params'](trial)
                model = model_info['model'](**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return -nse(y_test, y_pred)

            # Optimize model using Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=1000)

            # Best parameters
            best_params = study.best_params

            # Train the best model
            best_model = model_info['model'](**best_params)
            best_model.fit(X_train, y_train)

            # Predict on test data
            y_pred = best_model.predict(X_test)

            # Calculate Metrics
            nse_score = nse(y_test, y_pred)
            r2 = calculate_r2(y_test, y_pred)

            # Save predictions to CSV
            predictions_df = pd.DataFrame({'month': dates_test, 'TOC': y_test, 'Predicted-TOC': y_pred})
            predictions_output_path = f"{os.path.splitext(file_path)[0]}_{model_name}_(SYNTHETIC)_FINALS(E&S).csv"
            predictions_df.to_csv(predictions_output_path, index=False, encoding='utf-8-sig')

            # Save results
            results.append({
                'File': os.path.basename(file_path),
                'Model': model_name,
                'NSE': nse_score,
                'R2': r2,
                'Best Parameters': best_params
            })

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Compile results into a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file DOCHEON
output_path = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\SACHOEN_WGAN_SYNTHETIC-180_FINALS(E&S).csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')


print(f"Results saved to: {output_path}")



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# I SEPARATE THE HYBRID MODEL FROM THE OTHER ML ALGORITHM
# # M5 MODEL TREE
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import r2_score
# import optuna
# import os

# class M5ModelTree:
#     def __init__(self, min_samples_split=10, max_depth=5, min_samples_leaf=1, prune_alpha=0.01):
#         self.tree = DecisionTreeRegressor(
#             min_samples_split=min_samples_split, 
#             max_depth=max_depth,
#             min_samples_leaf=min_samples_leaf,
#             ccp_alpha=prune_alpha
#         )
#         self.linear_models = {}

#     def fit(self, X, y):
#         self.tree.fit(X, y)
#         leaf_indices = self.tree.apply(X)
#         for leaf in np.unique(leaf_indices):
#             indices = np.where(leaf_indices == leaf)
#             self.linear_models[leaf] = LinearRegression().fit(X.iloc[indices], y.iloc[indices])

#     def predict(self, X):
#         leaf_indices = self.tree.apply(X)
#         predictions = np.array([
#             self.linear_models[leaf].predict(X.iloc[[i]])[0] for i, leaf in enumerate(leaf_indices)
#         ])
#         return predictions

# # NSE Function
# def nse(y_true, y_pred):
#     denominator = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denominator == 0:
#         return np.nan
#     return 1 - (np.sum((y_true - y_pred) ** 2) / denominator)

# # R2 Calculation Function
# def calculate_r2(observed, predicted):
#     observed_mean = np.mean(observed)
#     predicted_mean = np.mean(predicted)
#     numerator = np.sum((observed - observed_mean) * (predicted - predicted_mean))
#     denominator = np.sqrt(np.sum((observed - observed_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2))
    
#     if denominator == 0:
#         return np.nan
#     return (numerator / denominator) ** 2

# # Optuna optimization function
# def objective(trial):
#     params = {
#         'max_depth': trial.suggest_int("max_depth", 3, 20),
#         'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 50),
#         'min_samples_split': trial.suggest_int("min_samples_split", 2, 50),
#         'prune_alpha': trial.suggest_float("prune_alpha", 1e-4, 0.1, log=True)
#     }
    
#     m5_tree = M5ModelTree(**params)
#     m5_tree.fit(X_train, y_train)
#     y_pred = m5_tree.predict(X_test)
#     return -nse(y_test, y_pred)

# # File Paths   SACHEON & DOCHEON (REAL DATA)
# file_paths = [
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_DATA_TRAIN_MONTHLY.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_DATA_TRAIN_MONTHLY.csv"
# ]


# # File Paths   SACHEON (WGAN DATA)
# file_paths = [
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-100-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-200-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-300-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-500-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1000-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1500-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-3000-Epochs.csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(180-Synthetic-data).csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(300-Synthetic-data).csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(600-Synthetic-data).csv"
# ]

# # File Paths   DOCHEON (WGAN DATA)
# file_paths = [
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-100-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-200-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-300-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1000-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-3000-Epochs.csv"
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(180-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(300-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(600-Synthetic-data).csv"
# ]

# # Features and Target
# features = ['prcp', 'tmax', 'tmin', 'rhum']
# target = 'TOC'
# date_column = 'month' 

# # Results List
# results = []

# # Loop through each file and apply models
# for file_path in file_paths:
#     try:
#         # Load Data
#         data = pd.read_csv(file_path)
#         X = data[features]
#         y = data[target]
#         dates = data[date_column]

#         # Impute missing values in X
#         imputer = SimpleImputer(strategy='mean')
#         X = pd.DataFrame(imputer.fit_transform(X), columns=features)

#         # Drop rows where the target variable is NaN
#         y = y.dropna()
#         X = X.loc[y.index]
#         dates = dates.loc[y.index]

#         # Split data into 80% training and 20% testing
#         X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
#             X, y, dates, test_size=0.017, random_state=42, shuffle=False
#         )

#         # Optimize model using Optuna
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=1000)
#         best_params = study.best_params

#         # Train the best model
#         best_m5_tree = M5ModelTree(**best_params)
#         best_m5_tree.fit(X_train, y_train)
#         y_pred_m5 = best_m5_tree.predict(X_test)

#         # Calculate Metrics
#         nse_score = nse(y_test, y_pred_m5)
#         r2 = calculate_r2(y_test, y_pred_m5)

#         # Save predictions to CSV
#         predictions_df = pd.DataFrame({'month': dates_test, 'TOC': y_test, 'Predicted-TOC': y_pred_m5})
#         predictions_output_path = f"{os.path.splitext(file_path)[0]}_SACHEON_WGAN_SYNTHETICS(M5).csv"       # Rename the files
#         predictions_df.to_csv(predictions_output_path, index=False, encoding='utf-8-sig')

#         # Save results
#         results.append({
#             'File': os.path.basename(file_path),
#             'Model': 'Optimized M5 Model Tree with Linear Regression',
#             'NSE': nse_score,
#             'R2': r2,
#             'Best Parameters': best_params
#         })

#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")

# # Compile results into a DataFrame
# results_df = pd.DataFrame(results)

# # Save results to a CSV file
# output_path = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\SACHEON_WGAN_SYNTHETIC_600(M5).csv" # SAVE and Rename the file to your directories
# results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# print(f"Results saved to: {output_path}")



# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# M5-SGB

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import r2_score
# from sklearn.ensemble import GradientBoostingRegressor
# import optuna
# import os

# class M5ModelTree:
#     def __init__(self, min_samples_split=10, max_depth=5, min_samples_leaf=1, prune_alpha=0.01):
#         self.tree = DecisionTreeRegressor(
#             min_samples_split=min_samples_split, 
#             max_depth=max_depth,
#             min_samples_leaf=min_samples_leaf,
#             ccp_alpha=prune_alpha
#         )
#         self.linear_models = {}

#     def fit(self, X, y):
#         self.tree.fit(X, y)
#         leaf_indices = self.tree.apply(X)
#         for leaf in np.unique(leaf_indices):
#             indices = np.where(leaf_indices == leaf)
#             self.linear_models[leaf] = LinearRegression().fit(X.iloc[indices], y.iloc[indices])

#     def predict(self, X):
#         leaf_indices = self.tree.apply(X)
#         predictions = np.array([
#             self.linear_models[leaf].predict(X.iloc[[i]])[0] for i, leaf in enumerate(leaf_indices)
#         ])
#         return predictions

# # Gradient Boosting Model Wrapper
# class GradientBoostedM5:
#     def __init__(self, base_model, learning_rate=0.1, n_estimators=100, max_depth=3):
#         self.base_model = base_model
#         self.boosting_model = GradientBoostingRegressor(
#             learning_rate=learning_rate, 
#             n_estimators=n_estimators, 
#             max_depth=max_depth
#         )

#     def fit(self, X, y):
#         # Fit M5 Tree first
#         self.base_model.fit(X, y)
#         base_predictions = self.base_model.predict(X)

#         # Compute residuals
#         residuals = y - base_predictions

#         # Train Gradient Boosting on residuals
#         self.boosting_model.fit(X, residuals)

#     def predict(self, X):
#         base_predictions = self.base_model.predict(X)
#         boost_predictions = self.boosting_model.predict(X)
#         return base_predictions + boost_predictions

# # NSE Function
# def nse(y_true, y_pred):
#     denominator = np.sum((y_true - np.mean(y_true)) ** 2)
#     if denominator == 0:
#         return np.nan
#     return 1 - (np.sum((y_true - y_pred) ** 2) / denominator)

# # R2 Calculation Function
# def calculate_r2(observed, predicted):
#     observed_mean = np.mean(observed)
#     predicted_mean = np.mean(predicted)
#     numerator = np.sum((observed - observed_mean) * (predicted - predicted_mean))
#     denominator = np.sqrt(np.sum((observed - observed_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2))
    
#     if denominator == 0:
#         return np.nan
#     return (numerator / denominator) ** 2

# # Optuna optimization function
# def objective(trial):
#     params = {
#         'max_depth': trial.suggest_int("max_depth", 3, 20),
#         'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 50),
#         'min_samples_split': trial.suggest_int("min_samples_split", 2, 50),
#         'prune_alpha': trial.suggest_float("prune_alpha", 1e-4, 0.1, log=True),
#         'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
#         'n_estimators': trial.suggest_int("n_estimators", 50, 500),
#         'boost_max_depth': trial.suggest_int("boost_max_depth", 2, 20)
#     }
    
#     m5_tree = M5ModelTree(
#         min_samples_split=params['min_samples_split'],
#         max_depth=params['max_depth'],
#         min_samples_leaf=params['min_samples_leaf'],
#         prune_alpha=params['prune_alpha']
#     )

#     boosted_model = GradientBoostedM5(
#         base_model=m5_tree,
#         learning_rate=params['learning_rate'],
#         n_estimators=params['n_estimators'],
#         max_depth=params['boost_max_depth']
#     )

#     boosted_model.fit(X_train, y_train)
#     y_pred = boosted_model.predict(X_test)
    
#     return -nse(y_test, y_pred)

# # File Paths   SACHEON & DOCHEON (REAL DATA)
# file_paths = [
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_DATA_TRAIN_MONTHLY.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_DATA_TRAIN_MONTHLY.csv"
# ]

# # File Paths   SACHEON (WGAN DATA)
# file_paths = [
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-100-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-200-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-300-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-500-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1000-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-1500-Epochs.csv",
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_EPOCHS_TOC-INPUT__MONTHLY\FINALLY-1-사천-TOC_GAN_data-3000-Epochs.csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(180-Synthetic-data).csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(300-Synthetic-data).csv"
#     r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천_WGAN_SYNTHETIC_TOC-INPUT__MONTHLY\1-사천-Training-Data_Monthly_Averages(600-Synthetic-data).csv"
# ]

# # File Paths   DOCHEON (WGAN DATA)
# file_paths = [
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-100-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-200-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-300-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1000-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-1500-Epochs.csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\DOCHEON_EPOCHS\FINALLY-2-도천-TOC_GAN_data-3000-Epochs.csv"
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(180-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(300-Synthetic-data).csv",
#     # r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(600-Synthetic-data).csv"
# ]


# # Features and Target
# features = ['prcp', 'tmax', 'tmin', 'rhum']
# target = 'TOC'
# date_column = 'month' 

# # Results List
# results = []

# # Loop through each file and apply models
# for file_path in file_paths:
#     try:
#         # Load Data
#         data = pd.read_csv(file_path)
#         X = data[features]
#         y = data[target]
#         dates = data[date_column]

#         # Impute missing values in X
#         imputer = SimpleImputer(strategy='mean')
#         X = pd.DataFrame(imputer.fit_transform(X), columns=features)

#         # Drop rows where the target variable is NaN
#         y = y.dropna()
#         X = X.loc[y.index]
#         dates = dates.loc[y.index]

#         # Split data into 80% training and 20% testing
#         X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
#             X, y, dates, test_size=0.017, random_state=42, shuffle=False
#         )

#         # Optimize model using Optuna
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=1000)
#         best_params = study.best_params

#         # Train the best boosted model
#         best_m5_tree = M5ModelTree(
#             min_samples_split=best_params['min_samples_split'],
#             max_depth=best_params['max_depth'],
#             min_samples_leaf=best_params['min_samples_leaf'],
#             prune_alpha=best_params['prune_alpha']
#         )

#         best_boosted_model = GradientBoostedM5(
#             base_model=best_m5_tree,
#             learning_rate=best_params['learning_rate'],
#             n_estimators=best_params['n_estimators'],
#             max_depth=best_params['boost_max_depth']
#         )

#         best_boosted_model.fit(X_train, y_train)
#         y_pred_m5 = best_boosted_model.predict(X_test)

#         # Calculate Metrics
#         nse_score = nse(y_test, y_pred_m5)
#         r2 = calculate_r2(y_test, y_pred_m5)

#         # Save predictions to CSV
#         predictions_df = pd.DataFrame({'month': dates_test, 'TOC': y_test, 'Predicted-TOC': y_pred_m5})
#         predictions_output_path = f"{os.path.splitext(file_path)[0]}_M5_SGB(SYNTHETIC_DOCHEON).csv"      #Rename the Files
#         predictions_df.to_csv(predictions_output_path, index=False, encoding='utf-8-sig')

#         # Save results
#         results.append({
#             'File': os.path.basename(file_path),
#             'Model': 'Boosted M5 Model Tree with Gradient Boosting',
#             'NSE': nse_score,
#             'R2': r2,
#             'Best Parameters': best_params
#         })

#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")

# # Compile results into a DataFrame
# results_df = pd.DataFrame(results)

# # Save results to a CSV file
# output_path = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\DOCHEON_WGAN_SYNTHETIC_600(M5-SGB).csv"  # SAVE and Rename the file to your directories
# results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# print(f"Results saved to: {output_path}")