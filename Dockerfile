FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# The trained model is a build output, not source: run `make train` and copy
# or mount artifacts/ in. Without it, /health reports "degraded" and /score
# returns 503 rather than the container failing.
RUN mkdir -p ./artifacts
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "crediwise.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
