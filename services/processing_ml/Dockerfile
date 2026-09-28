FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY services ./services
COPY libs ./libs
COPY pipelines ./pipelines

RUN uv sync --frozen --no-dev --package processing-ml

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["uv", "run", "--no-dev", "--package", "processing-ml"]
CMD ["admo-train"]
