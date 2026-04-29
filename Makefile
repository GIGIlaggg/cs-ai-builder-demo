.PHONY: install run demo eval test lint clean

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload --port 8000

demo:
	streamlit run streamlit_demo.py

eval:
	python -m app.eval

test:
	pytest -q

lint:
	ruff check app tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
