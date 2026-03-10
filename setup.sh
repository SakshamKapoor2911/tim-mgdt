#!/bin/bash
# setup.sh - Research Environment Initializer
echo "Initializing Research Environment..."

# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Additional research tools if needed
# pip install git+https://github.com/neelnanda-io/TransformerLens.git

echo "Environment Setup Complete. You can now run: python run_benchmarks.py"
