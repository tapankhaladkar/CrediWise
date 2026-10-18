.PHONY: install data train test serve notebook docker clean

install:
	pip install -r requirements-dev.txt && pip install -e .

# Re-download the dataset and verify its checksum. Not needed for a normal
# run: data/credit_default.csv is already in the repository.
data:
	python scripts/download_data.py

train:
	PYTHONPATH=src python -u -m crediwise.train
	python scripts/make_model_card.py

test:
	PYTHONPATH=src pytest -q

serve:
	PYTHONPATH=src uvicorn crediwise.api.main:app --reload --port 8000

# Re-run the walkthrough top to bottom and save its outputs.
notebook:
	PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

docker:
	docker build -t crediwise:latest .

clean:
	rm -rf artifacts/*.joblib reports/*.json .pytest_cache
