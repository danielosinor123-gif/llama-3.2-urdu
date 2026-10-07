#!/usr/bin/env bash
set -euo pipefail

echo "Pulling the base model..."
ollama pull llama3.2:3b-instruct

echo "Creating a local Urdu model alias..."
ollama create urdu-llama -f ollama/Modelfile

echo "Model is ready. Run it with:"
echo "ollama run urdu-llama"
