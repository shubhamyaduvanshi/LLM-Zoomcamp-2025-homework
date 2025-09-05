from datetime import datetime
import pandas as pd
import batch

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_result = batch.prepare_data(df, categorical)
    assert len(actual_result) == 2

    assert actual_result['duration'].between(1, 60).all()

    assert not actual_result[categorical].isna().any().any()

    for col in categorical:
        assert actual_result[col].dtype == object or str(actual_result[col].dtype) == 'string'
