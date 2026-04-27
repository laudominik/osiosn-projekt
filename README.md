

# Environment preparation

```
uv sync
uv pip install -e . 
```

# Monitoring runs

```
uv run aim up --port 4321
```

# Running all experiments

```
./scripts/run_all_experiments
```

Also generates all tables and figures


# Building report

```
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex 
```


claude --resume 3444f77f-89c2-43dd-ac4e-1caaa878407a