# flash-model-cache-bench
#
#   make setup    install pinned deps (uv), create .env, check credentials
#   make run      deploy -> benchmark -> report -> tear down (always)
#   make report   re-render the latest (or RUN=<id>) report
#   make billing  fetch billed GPU-minutes/cost for the latest (or RUN=<id>) run (~1 h after)
#   make clean    tear down anything a crashed/interrupted run left behind
#   make test     unit tests (no Runpod, no spend)
#
# Knobs are env vars (see .env.example), e.g. `make run MODEL=Qwen/Qwen2.5-7B-Instruct COLD_TRIALS=5`.

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help setup run report billing clean test

# Load .env into every recipe; command-line VAR=... still wins.
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif
export MODEL GPU DATACENTER VOLUME_GB STRATEGIES COLD_TRIALS WARM_PER_COLD MAX_NEW_TOKENS PARALLEL EVICT_PAGE_CACHE KEEP RUN_ID

help:
	@sed -n '3,10p' Makefile | sed 's/^# \{0,1\}//'

setup:
	@command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	uv sync
	@[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }
	@uv run python -c "from bench.runpod_api import api_key; api_key(); print('Runpod API key: found')" \
		|| { echo "-> put RUNPOD_API_KEY in .env, or run: uv run rp login"; exit 1; }
	@echo "setup ok — next: make run"

run:
	uv run python -m bench run

report:
	uv run python -m bench report $(RUN)

billing:
	uv run python -m bench billing $(RUN)

clean:
	uv run python -m bench clean

test:
	uv run pytest -q
