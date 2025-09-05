from datetime import datetime
import pandas as pd
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def get_input_path(year, month):
    # Use S3 for integration testing
    default_input_pattern = 's3://nyc-duration/test-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def test_save_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    input_file = get_input_path(2023, 1)
    print(f"Saving test data to: {input_file}")

    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    print("Test data saved successfully!")

    # Set environment variables for the batch script
    os.environ['S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
    os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/test-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    exit_code = os.system("python batch.py 2023 01")
    assert exit_code == 0

def test_read_output():
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    output_file = get_output_path(year=2023, month=1)
    print(f"Reading output from: {output_file}")

    df = pd.read_parquet(output_file, storage_options=options)
    total = df['predicted_duration'].sum()
    print(f"Total predicted duration: {total}")
    assert round(total, 2) == 36.28

if __name__ == '__main__':
    test_save_data()
    test_read_output()
