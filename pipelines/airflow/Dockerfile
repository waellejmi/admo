FROM apache/airflow:3.3.2-python3.13

USER root

RUN apt-get update \
    && apt-get install --no-install-recommends -y docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY libs/persistence /opt/admo/libs/persistence
USER airflow

RUN pip install --no-cache-dir --no-deps /opt/admo/libs/persistence
