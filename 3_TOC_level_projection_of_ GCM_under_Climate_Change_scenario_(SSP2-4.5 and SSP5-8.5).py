#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SACHEON RESERVOIR BEST MODEL (SVR) compared to other ML Algorithm
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Define file paths
train_file = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천-Training-Data_Monthly_Averages(180-Synthetic-data).csv"   #BEST WGAN Training Strategies

# List of multiple test files
test_files = [
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\11-ID4383_SQM_ACCESS-CM2_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\11-ID4383_SQM_ACCESS-CM2_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\22-ID4383_SQM_ACCESS-ESM1-5_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\22-ID4383_SQM_ACCESS-ESM1-5_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\33-ID4383_SQM_CanESM5_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\33-ID4383_SQM_CanESM5_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\44-ID4383_SQM_CMCC-ESM2_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\44-ID4383_SQM_CMCC-ESM2_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\55-ID4383_SQM_TaiESM1_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\55-ID4383_SQM_TaiESM1_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\66-ID4383_SQM_NorESM2-MM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\66-ID4383_SQM_NorESM2-MM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\77-ID4383_SQM_NorESM2-LM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\77-ID4383_SQM_NorESM2-LM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\88-ID4383_SQM_NESM3_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\88-ID4383_SQM_NESM3_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\99-ID4383_SQM_MRI-ESM2-0_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\99-ID4383_SQM_MRI-ESM2-0_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\100-ID4383_SQM_MIROC6_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\100-ID4383_SQM_MIROC6_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\111-ID4383_SQM_KIOST-ESM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\111-ID4383_SQM_KIOST-ESM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\112-ID4383_SQM_FGOALS-g3_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\112-ID4383_SQM_FGOALS-g3_ssp585_summary.csv"
]

# Features and target
features = ['prcp', 'tmax', 'tmin','rhum']
# features = ['prcp', 'tmax', 'tmin']
target = 'TOC'

# Load training data
train_df = pd.read_csv(train_file, parse_dates=['month'])
print(f"Training data loaded with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")

# Handle any missing values (dropping rows for simplicity)
train_df = train_df.dropna(subset=features + [target])

# Split features and target
X_train = train_df[features]
y_train = train_df[target]

# Standardize the features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Support Vector Regression model with specified parameters
svr_params = {'C':  32.55895169997554, 'epsilon':  0.24672844668472774}
model = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'])
model.fit(X_train_scaled, y_train)
print(f"Support Vector Regression model trained successfully with parameters: {svr_params}")

# Save the trained model and scaler
model_path = 'svr_model.joblib'
scaler_path = 'scaler.joblib'
dump(model, model_path)
dump(scaler, scaler_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")

# Process multiple test files
for test_file in test_files:
    # Load testing data
    test_df = pd.read_csv(test_file, parse_dates=['month'])
    print(f"Testing data loaded with {test_df.shape[0]} rows and {test_df.shape[1]} columns.")

    # Handle any missing values in testing data
    test_df = test_df.dropna(subset=features)

    # Combine year and month into a single column
    test_df['year_month'] = test_df['year'].astype(str) + "-" + test_df['month'].astype(str).str.zfill(2)

    # Prepare features for prediction
    X_test = test_df[features]
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Create a new DataFrame with date, year_month, and Predicted_TOC
    results_df = pd.DataFrame({
        'year_month': test_df['year_month'],
        'Predicted_TOC': predictions
    })

    # Define the output file path dynamically
    output_file = test_file.replace("2-TESTING DATA", "3-PREDICTIONS").replace(".csv", "_predictions.csv")

    # Save the results to the new CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")






#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#DOCHEON RESERVOIR BEST ML MODEL (M5-SGB) compared to other ML Algorithm
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import os
from sklearn.preprocessing import StandardScaler
from joblib import dump

class M5ModelTree:
    def __init__(self, min_samples_split=10, max_depth=5, min_samples_leaf=1, prune_alpha=0.01):
        self.tree = DecisionTreeRegressor(
            min_samples_split=min_samples_split, 
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=prune_alpha
        )
        self.linear_models = {}

    def fit(self, X, y):
        self.tree.fit(X, y)
        leaf_indices = self.tree.apply(X)
        for leaf in np.unique(leaf_indices):
            indices = np.where(leaf_indices == leaf)
            self.linear_models[leaf] = LinearRegression().fit(X.iloc[indices], y.iloc[indices])

    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        predictions = np.array([
            self.linear_models[leaf].predict(X.iloc[[i]])[0] for i, leaf in enumerate(leaf_indices)
        ])
        return predictions


# Gradient Boosting Model Wrapper
class GradientBoostedM5:
    def __init__(self, base_model, learning_rate=0.10568017185527617, n_estimators=302, max_depth=10):
        self.base_model = base_model
        self.boosting_model = GradientBoostingRegressor(
            learning_rate=learning_rate, 
            n_estimators=n_estimators, 
            max_depth=max_depth
        )

    def fit(self, X, y):
        # Fit M5 Model Tree first
        self.base_model.fit(X, y)
        base_predictions = self.base_model.predict(X)

        # Compute residuals
        residuals = y - base_predictions

        # Train Gradient Boosting on residuals
        self.boosting_model.fit(X, residuals)

    def predict(self, X):
        base_predictions = self.base_model.predict(X)
        boost_predictions = self.boosting_model.predict(X)
        return base_predictions + boost_predictions


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


# File Paths
# train_file = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\1-INPUT DATA\1-사천-Training-Data_Monthly_Averages(180-Synthetic-data).csv"
train_file = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-도천_GAN-DATA_TRAIN_MONTHLY(300-Synthetic-data).csv"

# List of multiple test files
test_files = [
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\11-ID4981_SQM_ACCESS-CM2_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\11-ID4981_SQM_ACCESS-CM2_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\22-ID4981_SQM_ACCESS-ESM1-5_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\22-ID4981_SQM_ACCESS-ESM1-5_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\33-ID4981_SQM_CanESM5_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\33-ID4981_SQM_CanESM5_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\44-ID4981_SQM_CMCC-ESM2_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\44-ID4981_SQM_CMCC-ESM2_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\55-ID4981_SQM_TaiESM1_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\55-ID4981_SQM_TaiESM1_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\66-ID4981_SQM_NorESM2-MM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\66-ID4981_SQM_NorESM2-MM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\77-ID4981_SQM_NorESM2-LM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\77-ID4981_SQM_NorESM2-LM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\88-ID4981_SQM_NESM3_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\88-ID4981_SQM_NESM3_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\99-ID4981_SQM_MRI-ESM2-0_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\99-ID4981_SQM_MRI-ESM2-0_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\100-ID4981_SQM_MIROC6_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\100-ID4981_SQM_MIROC6_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\111-ID4981_SQM_KIOST-ESM_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\111-ID4981_SQM_KIOST-ESM_ssp585_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\112-ID4981_SQM_FGOALS-g3_ssp245_summary.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\112-ID4981_SQM_FGOALS-g3_ssp585_summary.csv"
]

# Features and target
# features = ['prcp', 'tmax', 'tmin', 'rhum']
features = ['prcp', 'tmax', 'tmin']
target = 'TOC'
 
# Load training data
train_df = pd.read_csv(train_file, parse_dates=['month'])
print(f"Training data loaded with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")

# Handle missing values
train_df = train_df.dropna(subset=features + [target])

# Split features and target
X_train = train_df[features]
y_train = train_df[target]

# Train M5 Model Tree with optimized parameters
m5_params = {
    'max_depth': 5,
    'min_samples_leaf': 48,
    'min_samples_split': 31,
    'prune_alpha':  0.0006502709575476228
}
best_m5_tree = M5ModelTree(**m5_params)

# Train the boosted M5 Model Tree
boosted_m5_model = GradientBoostedM5(
    base_model=best_m5_tree,
    learning_rate=0.2379631932019388,
    n_estimators=79,
    max_depth=19
)

boosted_m5_model.fit(X_train, y_train)
print(f"Optimized M5 Model Tree with Gradient Boosting trained successfully.")


# Process multiple test files
for test_file in test_files:
    # Load testing data
    test_df = pd.read_csv(test_file, parse_dates=['month'])
    print(f"Testing data loaded with {test_df.shape[0]} rows and {test_df.shape[1]} columns.")

    # Handle missing values
    test_df = test_df.dropna(subset=features)

    # Combine year and month into a single column
    test_df['year_month'] = test_df['year'].astype(str) + "-" + test_df['month'].astype(str).str.zfill(2)

    # # Combine year and month into a single column
    # test_df['Date'] = test_df['date'].astype(str).str.zfill(2)

    # Prepare features for prediction
    X_test = test_df[features]
    
    # Make predictions
    predictions = boosted_m5_model.predict(X_test)

    # Calculate performance metrics
    if target in test_df.columns:
        y_test = test_df[target]
        nse_score = nse(y_test, predictions)
        r2_score_value = calculate_r2(y_test, predictions)
        print(f"NSE: {nse_score}, R2 Score: {r2_score_value}")


        # Create a new DataFrame with date, year_month, and Predicted_TOC
    results_df = pd.DataFrame({
        'year_month': test_df['year_month'],
        'Predicted_TOC': predictions
    })

    # Define output file path dynamically
    output_file = test_file.replace("2-TESTING DATA", "3-PREDICTIONS").replace(".csv", "_predictions(2-도천-M5-SGB).csv")
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
