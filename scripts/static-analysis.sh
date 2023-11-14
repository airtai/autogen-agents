#!/bin/bash
set -e

echo "Running mypy..."
mypy autogen_agents tests

echo "Running bandit..."
bandit -c pyproject.toml -r autogen_agents

echo "Running semgrep..."
semgrep scan --config auto --error
