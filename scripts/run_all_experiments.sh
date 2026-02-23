#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo "  ŚmieciNet Experiment Pipeline"
echo "========================================"

echo -e "\n[1/6] Etap 4 - Baseline"
uv run scripts/general/baseline.py

echo -e "\n[2/6] Etap 5 - Unstructured pruning"
uv run scripts/pruning/unstructured_oneshot.py
uv run scripts/pruning/unstructured_scheduled.py

echo -e "\n[3/6] Etap 5 - Structured pruning"
uv run scripts/pruning/structured_oneshot.py
uv run scripts/pruning/structured_scheduled.py

# Etap 6 – quantization
# echo -e "\n[4/6] Etap 6 - Post-training quantization"
# uv run python scripts/train_quantized_ptq.py

# echo -e "\n[4b/6] Etap 6 - Pruning + QAT"
# uv run python scripts/train_pruning_and_quant.py

# Etap 7 – hyperparameter search
# echo -e "\n[5/6] Etap 7 - Hyperparameter experiments"
# uv run python scripts/hyperopt.py all

echo -e "\n[6/6] Generating LaTeX tables and plots"
uv run python scripts/generate_tables.py
uv run python scripts/generate_plots.py

echo -e "\n========================================"
echo "  All experiments complete!"
echo "  Compile the report with:"
echo "    cd report && pdflatex main.tex"
echo "========================================"
