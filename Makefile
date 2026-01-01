PYTHON ?= python

.PHONY: setup test dataset train-classify train-regress tune report

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

dataset:
	$(PYTHON) -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a_sample.csv

train-classify:
	$(PYTHON) -m ns8lab.cli train --task task_a_view --samples 400 --seed 0

train-regress:
	$(PYTHON) -m ns8lab.cli train --task task_c_regress_n --samples 400 --seed 0

tune:
	$(PYTHON) -m ns8lab.cli tune --config configs/task_a.yaml

report: test
	@echo "Review reports/experiments and reports/figures for latest artifacts."
