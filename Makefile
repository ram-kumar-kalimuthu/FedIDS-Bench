.PHONY: install test smoke lint clean

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

smoke:
	python scripts/run_smoke.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache results/
