FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Artifacts are build outputs, not source. Run `make train` and mount or
# COPY them in; until one is present /health reports "degraded" and
# /score returns 503 rather than the container failing to build.
RUN mkdir -p ./artifacts

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "crediwise.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
