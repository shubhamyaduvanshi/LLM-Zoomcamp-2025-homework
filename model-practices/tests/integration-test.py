from datetime import datetime
import pandas as pd
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def get_input_path(year, month):
    # For integration testing, use S3 path instead of HTTPS
    default_input_pattern = 's3://nyc-duration/test-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

# Create test data
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

# Set up S3 options for LocalStack
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

# Write test data to S3 (LocalStack)
input_file = get_input_path(2023, 1)
print(f"Writing test data to: {input_file}")

try:
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    print("Test data written successfully!")

    # Verify we can read it back
    df_read = pd.read_parquet(input_file, storage_options=options)
    print(f"Successfully read back {len(df_read)} rows")
    print(df_read.head())

except Exception as e:
    print(f"Error: {e}")
    print("Make sure LocalStack is running and the bucket exists:")
    print("aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration")
