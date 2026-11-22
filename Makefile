.PHONY: install test test-e2e lint format push

install:  ## Create/update the virtual environment from uv.lock
	uv sync

test:  ## Run the test suite (what "main always runs" means; e2e tests excluded)
	uv run pytest

test-e2e:  ## Run the slow end-to-end tests (whole grid on a tiny model and fake data)
	uv run pytest -m e2e

lint:  ## Check style (recommended, not required for pushing)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix style
	uv run ruff format .
	uv run ruff check --fix .

push: test  ## Run the tests, then push the current branch
	git push -u origin HEAD
