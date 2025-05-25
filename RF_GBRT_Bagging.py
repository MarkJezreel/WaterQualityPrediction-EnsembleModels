from time import time, sleep
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
                             explained_variance_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(data, cols, title):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(title)
    plt.show()


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

    # Plot the distribution of raw data
    plot_distributions(data, cols, title='Raw Data Distribution')

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    data[cols] = scaler.fit_transform(data[cols])

    # Plot the distribution of normalized data
    plot_distributions(pd.DataFrame(data, columns=cols), cols, title='Normalized Data Distribution')

    # Plot the heatmap
    plot_heatmap(pd.DataFrame(data, columns=cols), title='Heatmap of Normalized Data')

    data.info()

    # Split the data into training, validation, and test sets
    train_data = data[data['Year'].between(2013, 2018)]
    validation_data = data[data['Year'].between(2019, 2021)]
    test_data = data[data['Year'].between(2022, 2023)]

    return train_data, validation_data, test_data, cols


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # Initialize base models
    rf_model = RandomForestRegressor()
    gb_model = GradientBoostingRegressor()

    # Create Bagging Regressors
    bagging_rf_model = BaggingRegressor(estimator=rf_model)
    bagging_gb_model = BaggingRegressor(estimator=gb_model)

    # Create the Voting Regressor
    voting_model = VotingRegressor(estimators=[('rf', bagging_rf_model), ('gbrt', bagging_gb_model)])

    # Measure training time
    start_train_time = time()
    voting_model.fit(X_train, y_train)
    train_time = time() - start_train_time

    # Measure prediction time
    start_pred_time = time()
    y_pred = voting_model.predict(X_val)
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

    # Output evaluation metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("Training Time (seconds):", train_time)
    print("Prediction Time (seconds):", prediction_time)

    return voting_model


def main():
    # start time
    start_time = time()
    # Introduction text
    print("Stacking Ensemble of Random Forest and Gradient Boosted Regression Trees\n")
    sleep(5)

    # Load and preprocess the data
    data_file = "Water_Quality_Dataset.csv"
    train_data, val_data, test_data, feature_cols = load_and_preprocess_data(data_file)
    print("The data is processing...")
    sleep(4)

    # Define features and target columns
    X_train = train_data[['Year', 'Month']]
    X_val = val_data[['Year', 'Month']]
    X_test = test_data[['Year', 'Month']]
    target_cols = ['Water Temperature', 'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']

    for target in target_cols:
        sleep(1)
        print(f"\nTraining model for {target}...")
        sleep(5)
        y_train = train_data[target]
        y_val = val_data[target]

        # Train and evaluate
        print(f"Evaluation for {target}:")
        model = train_and_evaluate(X_train, y_train, X_val, y_val)

        # Test set evaluation
        y_test = test_data[target]
        y_test_pred = model.predict(X_test)
        test_metrics = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
            'msle': mean_squared_log_error(y_test, y_test_pred),
            'explained_variance': explained_variance_score(y_test, y_test_pred),
        }

        # Output test set evaluation metrics
        print("\nTest Set Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value}")

    # End time
    end_time = time() - start_time
    print("\nTotal Execution Time (seconds):", end_time)


if __name__ == "__main__":
    main()
