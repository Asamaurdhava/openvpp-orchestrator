VENV ?= .venv
PY = $(VENV)/bin/python

.PHONY: help install data forecasters demo backtest deck dash clean reproduce

help:
	@echo "make install      install deps into $(VENV)"
	@echo "make data         generate synthetic fleet + 1-year signal traces"
	@echo "make forecasters  run forecaster sanity + visual demo"
	@echo "make demo         single-vehicle MILP demo"
	@echo "make backtest     full backtest (10 vehicles × 1 year)"
	@echo "make deck         regenerate deck.pptx from backtest results"
	@echo "make dash         launch Streamlit dashboard"
	@echo "make reproduce    data + backtest + deck (full end-to-end)"
	@echo "make clean        delete generated data and results"

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.fleet.generator
	$(PY) -m src.signals.grid_events
	$(PY) -m src.signals.lmp
	$(PY) -m src.signals.inference_demand
	$(PY) -m src.sanity_check

forecasters:
	$(PY) -m src.forecasting.demo

demo:
	$(PY) -m src.optimizer.demo

backtest:
	$(PY) -m src.simulator.backtest --n-vehicles 10 --horizon-days 365

deck:
	$(PY) -m src.pitch.deck

dash:
	$(VENV)/bin/streamlit run src/dashboard/app.py

reproduce: data backtest deck

clean:
	rm -f data/*.parquet
	rm -f results/*.csv results/*.parquet results/*.png results/*.html
	rm -f deck.pptx
