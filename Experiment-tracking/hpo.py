import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

HPO_EXPERIMENT_NAME="random-forest-hyperopt"
mlflow.set_tracking_uri("http://127.0.0.1:5050")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    print("Run_optimization started .....")
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        print("mlflow server started nested....")
        with mlflow.start_run(nested=True): 
            rf = RandomForestRegressor(**params)
            mlflow.log_params(params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            print(f'rmse found is {rmse}')
            mlflow.log_metric("rmse", rmse)
            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )
    # client = MlflowClient()
    try:
        client = MlflowClient()
        print(f"Connected to MLflow server at {client.tracking_uri}")
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")
        raise
    
    print("mlflow client started")
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )
    print (f"Tuning model completed and best run is {best_run}")


if __name__ == '__main__':
    run_optimization()