import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def save_results(
    experiment_id: str, metrics: dict, config: dict = None, extra: dict = None
):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{experiment_id}.json"
    data = {
        "id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
        "metrics": metrics,
    }
    if extra:
        data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[results] Saved → {path}")
    return str(path)


def load_results(experiment_id: str) -> dict:
    path = RESULTS_DIR / f"{experiment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{experiment_id}' at {path}")
    with open(path) as f:
        return json.load(f)


def load_example_results() -> dict:
    path = RESULTS_DIR / "example_results.json"
    with open(path) as f:
        return json.load(f)


def extract_trainer_metrics(trainer) -> dict:
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    return metrics
