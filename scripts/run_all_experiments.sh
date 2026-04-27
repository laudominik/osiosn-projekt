#!/usr/bin/env bash
# Run the full experiment pipeline in sequence.
# Usage: bash run_all_experiments.sh
#
# Prerequisites: activate the uv/venv environment first:
#   source .venv/bin/activate  OR  uv run <script>

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo "  OSiOSN Experiment Pipeline"
echo "========================================"

# Etap 4 – baseline
echo -e "\n[1/6] Etap 4 - Baseline"
uv run python scripts/train_baseline.py

# Etap 5 – pruning
echo -e "\n[2/6] Etap 5 - Unstructured pruning"
uv run python scripts/train_pruning_unstructured.py

echo -e "\n[3/6] Etap 5 - Structured pruning"
uv run python scripts/train_pruning_structured.py

# Etap 6 – quantization
echo -e "\n[4/6] Etap 6 - Post-training quantization"
uv run python scripts/train_quantized_ptq.py

echo -e "\n[4b/6] Etap 6 - Pruning + QAT"
uv run python scripts/train_pruning_and_quant.py

# Etap 7 – hyperparameter search
echo -e "\n[5/6] Etap 7 - Hyperparameter experiments"
uv run python scripts/train_etap7.py all

# Generate report assets
echo -e "\n[6/6] Generating LaTeX tables and plots"
uv run python scripts/generate_tables.py
uv run python scripts/generate_plots.py

echo -e "\n========================================"
echo "  All experiments complete!"
echo "  Compile the report with:"
echo "    cd report && pdflatex main.tex"
echo "========================================"
