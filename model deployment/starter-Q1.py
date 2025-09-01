import pickle
import pandas as pd
import sys
import os
from prefect import flow, task
from google.cloud import storage

with open("model.bin", 'rb') as f_in:
        dv, model = pickle.load(f_in)

@task
def load_data(filename):
    df = pd.read_parquet(filename)
    return df

@task
def transform_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def prepare_data(df, categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return X_val

@task
def apply_model(X_val):
    y_pred = model.predict(X_val)
    print(f"The mean predicted duration is {round(y_pred.mean(), 3)} minutes")
    return y_pred

@task
def make_result(df, y_pred):
    year_df = df['tpep_pickup_datetime'].dt.year.astype(str).str.zfill(4)
    month_df = df['tpep_pickup_datetime'].dt.month.astype(str).str.zfill(2)

    df['ride_id'] = year_df + '/' + month_df + '_' + df.index.astype(str)

    df_result = df[["ride_id"]].copy()
    df_result['duration'] = y_pred
    return df_result

@task
def save_result(df_result, output_folder, year, month):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = f"{output_folder}/result_yellow_tripdata_{year}-{month}.parquet"
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return

def upload_blob(project_id, bucket_name, source_file_name, destination_blob_name):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded as {destination_blob_name}")
    return

@task
def upload2cloud(project_id, bucket_name, output_folder, year, month):
    filename = f"{output_folder}/result_yellow_tripdata_{year}-{month}.parquet"
    upload_blob(
        project_id=project_id,
        bucket_name=bucket_name,
        source_file_name=filename,
        destination_blob_name=filename
    )
    return

@flow(name="Taxi ML Pipeline", retries=1, retry_delay_seconds=300)
def taxi_pipeline(project_id, bucket_name, year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    output_folder = f"output"
    categorical = ['PULocationID', 'DOLocationID']

    df = load_data(filename)
    df = transform_data(df, categorical)
    X_val = prepare_data(df, categorical)
    y_pred = apply_model(X_val)
    # df_result = make_result(df, y_pred)
    # save_result(df_result, output_folder, year, month)
    # upload2cloud(project_id, bucket_name, output_folder, year, month)
    return

if __name__ == '__main__':
    year = sys.argv[1]
    month = sys.argv[2]
    project_id="your_project_id"
    bucket_name='your_bucket_name'
    taxi_pipeline(project_id, bucket_name, year, month)