.PHONY: help setup download inventory wf calendar climate soil master select cluster train evaluate all test clean

PY := python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

setup:     ## Install dependencies
	pip install -r requirements.txt

download:  ## Fetch the crop calendar; print manual steps for WF / climate / soil
	$(PY) scripts/00_download_data.py

inventory: ## Inventory every raw file before modelling
	$(PY) scripts/00_data_inventory.py

wf:        ## Stage 1 - water footprint target
	$(PY) scripts/01_prepare_wf.py

calendar:  ## Stage 2 - crop calendar + seven-day rule
	$(PY) scripts/02_prepare_calendar.py

climate:   ## Stage 3 - daily -> monthly climate (use PR_ONLY=1 for a dry run)
	$(PY) scripts/03_prepare_climate.py $(if $(PR_ONLY),--variable pr,)

soil:      ## Stage 4 - DSMW soil attributes
	$(PY) scripts/04_prepare_soil.py

master:    ## Stage 5 - integration + candidate feature table
	$(PY) scripts/05_build_master_dataset.py

select:    ## Stage 6 - clean, split, Pearson + XGBoost -> 20 features
	$(PY) scripts/06_select_features.py

cluster:   ## Stage 7 - elbow/silhouette, K-means + hierarchical (K=5)
	$(PY) scripts/07_cluster.py --target total_wf

train:     ## Stage 8 - cluster-wise AdaBoost
	$(PY) scripts/08_train_models.py --target total_wf

evaluate:  ## Stage 9 - metrics and figures
	$(PY) scripts/09_evaluate.py

all:       ## Run the whole pipeline
	$(PY) scripts/run_all.py

test:      ## Run unit tests
	pytest -q

clean:     ## Remove interim/processed artefacts (raw data untouched)
	rm -rf data/interim/* data/processed/* models/* results/predictions/*
	touch data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep results/predictions/.gitkeep
