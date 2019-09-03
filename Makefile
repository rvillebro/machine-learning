VENV=.venv

.PHONY: install, html

install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate; \
	pip install -r requirements.txt