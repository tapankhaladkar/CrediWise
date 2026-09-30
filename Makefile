.PHONY: install train train-nobureau test serve docker clean

install:
	pip install -r requirements-dev.txt && pip install -e .

train:
	PYTHONPATH=src python -u -m crediwise.train --demo-leakage
	python scripts/make_model_card.py

train-nobureau:
	PYTHONPATH=src python -u -m crediwise.train --no-bureau

test:
	PYTHONPATH=src pytest -q

serve:
	PYTHONPATH=src uvicorn crediwise.api.main:app --reload --port 8000

docker:
	docker build -t crediwise:latest .

clean:
	rm -rf artifacts/*.joblib reports/*.json reports/*.csv .pytest_cache
