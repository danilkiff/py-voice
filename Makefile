.PHONY: setup test

setup:
	git config core.hooksPath .githooks

test:
	uv run pytest
