from time import time, sleep
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor, \
    StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
                             explained_variance_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_rel


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Replace "N/A" with np.nan
    data.replace("N/A", np.nan, inplace=True)

    # Convert relevant columns to numeric, handle non-numeric values with 'coerce'
    cols = ['Water Temperature', 'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

    # Ensure "Year & Month" is in datetime format
    data["Year & Month"] = pd.to_datetime(data["Year & Month"], errors='coerce')

    # Drop rows where "Year & Month" could not be converted to datetime
    data.dropna(subset=["Year & Month"], inplace=True)

    # Extract useful numerical features from "Year & Month"
    data['Year'] = data['Year & Month'].dt.year
    data['Month'] = data['Year & Month'].dt.month

    # Impute missing values with the mean strategy
    imputer = SimpleImputer(strategy='mean')
    data[cols] = imputer.fit_transform(data[cols])

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    data[cols] = scaler.fit_transform(data[cols])

    data.info()

    # Split the data into training, validation, and test sets
    train_data = data[data['Year'].between(2013, 2018)]
    validation_data = data[data['Year'].between(2019, 2021)]
    test_data = data[data['Year'].between(2022, 2023)]

    return train_data, validation_data, test_data, cols


def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    # Measure training time
    start_train_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_train_time

    # Measure prediction time
    start_pred_time = time()
    y_pred = model.predict(X_val)
    prediction_time = time() - start_pred_time

    # Calculate evaluation metrics
    metrics = {
        'mse': mean_squared_error(y_val, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
        'mae': mean_absolute_error(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
        'msle': mean_squared_log_error(y_val, y_pred),
        'explained_variance': explained_variance_score(y_val, y_pred),
    }

    return metrics, train_time, prediction_time


def main():
    # Start time
    start_time = time()
    # Introduction text
    print("Comparing Bagging and Stacking Ensembles for Water Quality Prediction\n")
    sleep(5)

    # Load and preprocess the data
    data_file = "Water_Quality_Dataset.csv"
    train_data, val_data, test_data, feature_cols = load_and_preprocess_data(data_file)
    print("The data is processing...")
    sleep(4)

    # Define features and target columns
    X_train = train_data[['Year', 'Month']]
    X_val = val_data[['Year', 'Month']]
    target_cols = ['Water Temperature', 'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']

    voting_metrics = {}
    stacking_metrics = {}

    for target in target_cols:
        sleep(1)
        print(f"\nTraining models for {target}...")
        sleep(5)
        y_train = train_data[target]
        y_val = val_data[target]

        # Initialize models
        bagging_rf_model = BaggingRegressor(estimator=RandomForestRegressor())
        bagging_gb_model = BaggingRegressor(estimator=GradientBoostingRegressor())
        voting_model = VotingRegressor(estimators=[('rf', bagging_rf_model), ('gbrt', bagging_gb_model)])

        stacking_model = StackingRegressor(
            estimators=[('rf', RandomForestRegressor()), ('gbrt', GradientBoostingRegressor())],
            final_estimator=LinearRegression()
        )

        # Train and evaluate bagging models
        print(f"Evaluating Voting Model for {target}...")
        voting_metrics[target], voting_train_time, voting_pred_time = train_and_evaluate(voting_model, X_train, y_train,
                                                                                         X_val, y_val)

        # Train and evaluate stacking model
        print(f"Evaluating Stacking Model for {target}...")
        stacking_metrics[target], stacking_train_time, stacking_pred_time = train_and_evaluate(stacking_model, X_train,
                                                                                               y_train, X_val, y_val)

        # Print metrics
        print(f"\nVoting Model Metrics for {target}:")
        for metric_name, metric_value in voting_metrics[target].items():
            print(f"{metric_name}: {metric_value}")
        print("Training Time (seconds):", voting_train_time)
        print("Prediction Time (seconds):", voting_pred_time)

        print(f"\nStacking Model Metrics for {target}:")
        for metric_name, metric_value in stacking_metrics[target].items():
            print(f"{metric_name}: {metric_value}")
        print("Training Time (seconds):", stacking_train_time)
        print("Prediction Time (seconds):", stacking_pred_time)

    # Perform paired t-test on metrics
    for metric in ['mse', 'rmse', 'mae', 'r2', 'msle', 'explained_variance']:
        voting_scores = [voting_metrics[target][metric] for target in target_cols]
        stacking_scores = [stacking_metrics[target][metric] for target in target_cols]
        t_stat, p_val = ttest_rel(voting_scores, stacking_scores)
        print(f"\nPaired t-test for {metric}:")
        print(f"T-statistic: {t_stat}, P-value: {p_val}")

    # End time
    end_time = time() - start_time
    print("\nTotal Execution Time (seconds):", end_time)


if __name__ == "__main__":
    main()
