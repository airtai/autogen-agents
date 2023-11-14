#!/bin/bash

echo "Running pyup_dirs..."
pyup_dirs --py38-plus --recursive autogen_agents examples tests

echo "Running ruff..."
ruff autogen_agents examples tests --fix

echo "Running black..."
black autogen_agents examples tests
