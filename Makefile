# ─────────────────────────────────────────────────────────────────────────────
# Healthcare Predictive MLOps — Makefile
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help
SHELL         := /bin/bash

# ── Configurable defaults ────────────────────────────────────────────────────
TARGET        ?= dev          # override: make deploy TARGET=stg
PYTHON        ?= python3
CATALOG_NAME  ?= quickstart_catalog
SCHEMA_NAME   ?= healthcare_ml

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}' \
	  | sort

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install install-dev

install: ## Install production dependencies via uv
	uv sync --no-dev

install-dev: ## Install all dependencies (including dev extras) via uv
	uv sync --extra dev

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: lint format check

lint: ## Run ruff linter
	uv run ruff check src/ tests/ notebooks/

format: ## Auto-format code with ruff
	uv run ruff format src/ tests/ notebooks/

check: lint ## Alias for lint (used in CI)

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: test test-cov

test: ## Run unit tests
	uv run pytest tests/

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src/healthcare_mlops --cov-report=term-missing --cov-report=xml

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: upload-data

upload-data: ## Upload data/ to the Databricks Volume (CATALOG/SCHEMA overridable)
	bash scripts/upload_data.sh --catalog $(CATALOG_NAME) --schema $(SCHEMA_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Databricks Bundle — Validate
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: validate validate-dev validate-stg validate-prod

validate: ## Validate bundle for TARGET (default: dev)
	databricks bundle validate -t $(TARGET)

validate-dev: ## Validate bundle for dev
	databricks bundle validate -t dev

validate-stg: ## Validate bundle for stg
	databricks bundle validate -t stg

validate-prod: ## Validate bundle for prod
	databricks bundle validate -t prod

# ─────────────────────────────────────────────────────────────────────────────
# Databricks Bundle — Deploy
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: deploy deploy-dev deploy-stg deploy-prod

deploy: ## Deploy bundle to TARGET (default: dev)
	databricks bundle deploy -t $(TARGET)

deploy-dev: ## Deploy bundle to dev
	databricks bundle deploy -t dev

deploy-stg: ## Deploy bundle to stg
	databricks bundle deploy -t stg

deploy-prod: ## Deploy bundle to prod
	databricks bundle deploy -t prod

# ─────────────────────────────────────────────────────────────────────────────
# Databricks Bundle — Run jobs
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run run-pipeline run-feature-engineering run-training run-inference

run: ## Run ml_pipeline_workflow on TARGET (default: dev)
	databricks bundle run -t $(TARGET) ml_pipeline_workflow

run-pipeline: run ## Alias for run

run-feature-engineering: ## Run feature_engineering_job on TARGET
	databricks bundle run -t $(TARGET) feature_engineering_job

run-training: ## Run model_training_job on TARGET
	databricks bundle run -t $(TARGET) model_training_job

run-inference: ## Run batch_inference_job on TARGET
	databricks bundle run -t $(TARGET) batch_inference_job

# ─────────────────────────────────────────────────────────────────────────────
# Databricks Bundle — Destroy
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: destroy destroy-dev destroy-stg destroy-prod

destroy: ## Destroy bundle resources on TARGET (prompts for confirmation)
	@read -p "Destroy $(TARGET) resources? [y/N] " ans && [ "$$ans" = "y" ]
	databricks bundle destroy -t $(TARGET)

destroy-dev: ## Destroy dev resources (prompts for confirmation)
	@read -p "Destroy dev resources? [y/N] " ans && [ "$$ans" = "y" ]
	databricks bundle destroy -t dev

destroy-stg: ## Destroy stg resources (prompts for confirmation)
	@read -p "Destroy stg resources? [y/N] " ans && [ "$$ans" = "y" ]
	databricks bundle destroy -t stg

destroy-prod: ## Destroy prod resources (prompts for confirmation)
	@read -p "Destroy prod resources? [y/N] " ans && [ "$$ans" = "y" ]
	databricks bundle destroy -t prod

# ─────────────────────────────────────────────────────────────────────────────
# Compound targets
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: ci cd-dev cd-stg cd-prod

ci: install-dev check test validate-dev ## Full CI gate (lint + test + bundle validate)

cd-dev: install validate-dev deploy-dev ## Deploy to dev (CI/CD pipeline step)

cd-stg: install validate-stg deploy-stg ## Deploy to stg (CI/CD pipeline step)

cd-prod: install validate-prod deploy-prod ## Deploy to prod (CI/CD pipeline step)
