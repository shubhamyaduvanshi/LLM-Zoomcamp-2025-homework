from datetime import timedelta
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy import sparse
import mlflow
from prefect import flow, task
from prefect.client.orchestration import get_client
from pathlib import Path
import os

# Explicit orchestration setup for Prefect 3.x
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Debug API URL and Client
print(f"PREFECT_API_URL: {os.getenv('PREFECT_API_URL')}")
print("Connecting to Prefect Server...")

try:
    client = get_client()
    print(f"Connected to Prefect Server: {client.api_url}")
except Exception as e:
    print(f"Error connecting to Prefect Server: {e}")
    raise

# Define the data source URL
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"


@task
def load_data():
    """
    Load the Yellow Taxi data from the specified URL.
    """
    df = pd.read_parquet(url)
    print(f"Loaded {len(df):,} records")
    print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


@task
def transform_data(df):
    """
    Transform the raw data by calculating trip duration and filtering the dataset.
    """
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"Data after preparation: {len(df):,} rows")
    return df


@task
def prepare_data(df):
    """
    Prepare the data for machine learning by converting to dictionary and vectorizing.
    """
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    y = df["duration"].to_numpy()
    return X, y


@task
def train_model(X_train, y_train):
    """
    Train a Linear Regression model on the prepared data and log it using MLflow.
    """
    # Set MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("hw3-orchestration-nyc-taxi")

    with mlflow.start_run():
        # Train a Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Log model parameters and metrics
        mlflow.log_param("intercept_", lr.intercept_)
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="artifacts_local_orchestrator",
            registered_model_name="MyLinearRegressor_orchestrator"
        )
        print("Model training completed and logged to MLflow.")


@flow(name="Taxi ML Orchestrator", retries=1, retry_delay_seconds=300)
def taxi_pipeline():
    """
    Orchestrates the entire ML pipeline: data loading, transformation, preparation, and training.
    """
    print("Starting the Taxi ML Orchestrator pipeline!")
    df = load_data()
    df = transform_data(df)
    X, y = prepare_data(df)
    train_model(X_train=X, y_train=y)


if __name__ == "__main__":
    # Run the ML pipeline
    taxi_pipeline()