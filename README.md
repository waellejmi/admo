# ADMO - (Anomaly Detection MlOps Platform)

ADMO is an end-to-end MLOps platform for detecting anomalies in a large,
regularly updated French energy-consumption [dataset](https://data.ademe.fr/datasets/operat03-ratio-conso-ajustee) containing more than one
million records.

The project explores how to turn an experimental machine-learning workflow
into a modular, service-oriented system. The workflow is split into
independent Python packages for data ingestion, preprocessing, model training,
evaluation, and inference.

The main data flow is:

```text
ADEME CSV
  -> ingestion creates a raw Parquet snapshot
  -> processing creates a clean development dataset
  -> training fits and registers an Isolation Forest model
  -> evaluation measures the persisted model on labeled anomalies
  -> inference serves predictions through a FastAPI API
```

Garage provides S3-compatible object storage for artifacts, MLflow can track
experiments and production training runs, and Apache Airflow can orchestrate
the data, training, and evaluation workflows. Docker Compose provides the
local development stack; the packages are also designed to be used as
independent workloads.

## Project packages

| Package | Responsibility |
| --- | --- |
| `services/ingestion` | Converts source ADEME data into immutable raw Parquet artifacts. |
| `services/processing_ml` | Validates, preprocesses, samples, trains, evaluates, and creates labeled datasets. |
| `services/inference` | Serves predictions from persisted model artifacts through FastAPI. |
| `libs/persistence` | Provides PostgreSQL metadata and Garage object-storage integration. |
| `pipelines/airflow` | Defines scheduled and manually triggered workflow DAGs. |

## Getting started

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose for the local infrastructure

Install all workspace dependencies from the repository root:

```bash
uv sync
```

### Start local infrastructure

Start PostgreSQL, Garage, and MLflow:

```bash
docker compose \
  -f infra/database_setup-compose.yml \
  -f infra/docker-compose.mlflow.yml \
  up -d
```

Apply the persistence migrations before the first data or training run:

```bash
uv run --package admo-persistence alembic \
  -c libs/persistence/alembic.ini upgrade head
```

### Run the workloads

### Ingestion

Convert an ADEME CSV source into an immutable raw Parquet artifact and upload
it to Garage:

```bash
uv run --package ingestion admo-ingest \
  data/OPERAT03_RATIO_CONSO_AJUSTEE.csv \
  raw/ademe/version=<version>/source.parquet
```

### Processing and ML

Create a clean development dataset from the raw Parquet artifact:

```bash
uv run --package processing-ml admo-create-subdataset \
  --raw-key raw/ademe/version=<version>/source.parquet \
  --clean-key processed/ademe/version=<version>/clean_100k.parquet
```

Train the production Isolation Forest model:

```bash
uv run --package processing-ml admo-train \
  --training-key processed/ademe/version=<version>/clean_100k.parquet \
  --model-key models/isolation_forest/version=<version>/model.joblib \
  --preprocessor-key models/isolation_forest/version=<version>/preprocessor.joblib \
  --version <version>
```

Create an evaluation dataset and evaluate the persisted model with
`admo-inject-anomalies` and `admo-evaluate`. Run either command with `--help`
for its options.

### Inference API

Start the local FastAPI service:

```bash
uv run --package inference python -m admo_inference
```

The API provides `GET /health` and `POST /predict`.

### Airflow

Start Airflow locally with the repository DAGs:

```bash
uv run --package admo-airflow airflow standalone
```

Airflow requires the local
PostgreSQL, Garage, and MLflow services to be running and their connection
variables to be configured in the environment.

### Persistence

The persistence package is a shared library rather than a standalone service.
Use the Alembic command above to create or upgrade its PostgreSQL metadata
schema.

## Tests

Run the full test suite from the repository root:

```bash
uv run --all-packages pytest -q
```
