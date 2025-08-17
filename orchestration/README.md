# 🚕 Taxi Pipeline with Prefect 3 and MLflow 2

![Pipeline_run](./src/run.jfif)

This project demonstrates a data pipeline using [Prefect 3.4.5](https://docs.prefect.io/) for orchestration and [MLflow 2.22.1](https://mlflow.org/) for experiment tracking, all built with Python 3.10.

---

## ⚙️ Requirements

- Python 3.10
- `pip` or `virtualenv` / `venv`

---

## 📦 Installation

Create and activate a virtual environment (recommended):

```
python3.10 -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```
pip install prefect==3.4.5 mlflow==2.22.1 pandas scikit-learn

```

---

## 🚀 Start MLflow Tracking Server

Run the MLflow tracking server with a local SQLite backend and local artifact store:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local
```

MLflow UI available at: 📍 `http://localhost:5000`

Metadata will be stored in mlflow.db

Artifacts will be saved to the `./artifacts_local/` directory


> Ensure the artifacts_local folder exists or will be created automatically.

---

## 🧭 Start Prefect Server

To start the Prefect API and UI locally (Orion):

```
prefect server start
```

This will open the Prefect UI at:
📍 `http://localhost:4200`

> ⚠️ Make sure the Prefect server is running before running any flow that uses orchestration or scheduling.

---

## ▶️ Run the Pipeline

Once both the MLflow and Prefect servers are running, you can execute the pipeline:

```
python taxi_pipeline.py
```

This will:
- Run your Prefect flow.
- Track experiment data with MLflow.
- Optionally register tasks and flows in Prefect if using deployment or scheduling.

---

## ✅ Notes

Ensure the `MLFLOW_TRACKING_URI` is set correctly in your environment or inside `taxi_pipeline.py`.

For production use, consider configuring:
- MLflow with a backend store (PostgreSQL, MySQL)
- Artifact store (S3, MinIO, GCS)
- Prefect deployments for scheduling.



### List of available orchrestration tool in MLOPs are-
Airflow
Prefect
Dagster
Kestra
Mage
or some other tool