.PHONY: install test lint format push

install:  ## Create/update the virtual environment from uv.lock
	uv sync

test:  ## Run the whole test suite (what "main always runs" means)
	uv run pytest

lint:  ## Check style (recommended, not required for pushing)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix style
	uv run ruff format .
	uv run ruff check --fix .

push: test  ## Run the tests, then push the current branch
	git push -u origin HEAD
