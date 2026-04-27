"""
Generuje wykresy do raportu LaTeX z wynikami eksperymentów.

Czyta: results/example_results.json
Zapisuje: report/figures/*.pdf  (i *.png do podglądu)

Uruchomienie:
    uv run python scripts/generate_plots.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
COLORS = plt.cm.tab10.colors


def save_fig(name: str):
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"  ✓  {name}.pdf")
    plt.close()


def load_data() -> dict:
    with open(RESULTS_DIR / "example_results.json") as f:
        return json.load(f)


# ─── 1. Training curves ───────────────────────────────────────────────────────

def plot_training_curves(d: dict):
    h = d["baseline"]["training_history"]
    epochs = h["epochs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, h["train_loss"], "o-", color=COLORS[0], label="Trening")
    ax1.plot(epochs, h["val_loss"],   "s--", color=COLORS[1], label="Walidacja")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Strata (CrossEntropy)")
    ax1.set_title("Krzywe strat – model odniesienia")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [v * 100 for v in h["train_acc"]], "o-", color=COLORS[0], label="Trening (zaszumione etykiety)")
    ax2.plot(epochs, [v * 100 for v in h["val_acc"]],   "s--", color=COLORS[1], label="Walidacja (czyste etykiety)")
    ax2.axhline(d["baseline"]["metrics"]["test_acc"] * 100, color=COLORS[2],
                linestyle=":", linewidth=1.5, label=f"Test: {d['baseline']['metrics']['test_acc']*100:.1f}%")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Dokładność [%]")
    ax2.set_title("Krzywe dokładności – model odniesienia")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.suptitle("Etap 4 – douczanie modelu odniesienia (MobileNetV2 x0.75)", fontsize=13)
    plt.tight_layout()
    save_fig("training_curves")


# ─── 2. Pruning accuracy–sparsity tradeoff ────────────────────────────────────

def plot_pruning_tradeoff(d: dict):
    exps = d["pruning_experiments"]
    baseline_acc = d["baseline"]["metrics"]["test_acc"] * 100

    unstruct = [(e["sparsity"] * 100, e["metrics"]["test_acc"] * 100)
                for e in exps if e["method"] == "unstructured"]
    struct   = [(e["sparsity"] * 100, e["metrics"]["test_acc"] * 100)
                for e in exps if e["method"] == "structured"]

    fig, ax = plt.subplots(figsize=(8, 5))

    sp_u, ac_u = zip(*sorted(unstruct))
    sp_s, ac_s = zip(*sorted(struct))

    ax.plot([0] + list(sp_u), [baseline_acc] + list(ac_u),
            "o-", color=COLORS[0], label="Niestrukturalne L1")
    ax.plot([0] + list(sp_s), [baseline_acc] + list(ac_s),
            "s--", color=COLORS[1], label="Strukturalne LN")
    ax.axhline(baseline_acc, color="gray", linestyle=":", linewidth=1, label="Baseline (brak przerzedzania)")

    ax.set_xlabel("Rzadkość [%]")
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Etap 5 – wpływ przerzedzania na dokładność klasyfikacji")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    save_fig("pruning_accuracy_tradeoff")


# ─── 3. Model size comparison ─────────────────────────────────────────────────

def plot_model_size(d: dict):
    baseline_size = d["baseline"]["metrics"]["size_mb"]
    ptq  = d["quantization_experiments"][0]["metrics"]
    qat  = d["quantization_experiments"][1]["metrics"]
    pq   = d["quantization_experiments"][2]["metrics"]
    best = d["etap7_experiments"]["best_optimized"]["metrics"]

    labels = [
        "Baseline\n(FP32)",
        "PTQ INT8\n(Etap 6)",
        "QAT INT8\n(Etap 6)",
        "Pruning 50%\n+ PTQ",
        "Najlepszy\n+ Pruning 30%\n+ PTQ",
    ]
    sizes = [baseline_size, ptq["size_mb"], qat["size_mb"], pq["size_mb"], best["size_mb"]]
    colors_bar = [COLORS[0], COLORS[1], COLORS[1], COLORS[2], COLORS[3]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, sizes, color=colors_bar, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f} MB", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Rozmiar modelu [MB]")
    ax.set_title("Etap 6 – porównanie rozmiarów modelu po różnych optymalizacjach")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, baseline_size * 1.25)

    plt.tight_layout()
    save_fig("model_size_comparison")


# ─── 4. Inference time comparison ────────────────────────────────────────────

def plot_inference_time(d: dict):
    baseline_cpu = d["baseline"]["metrics"]["inference_cpu_ms"]

    struct_exps = [e for e in d["pruning_experiments"] if e["method"] == "structured"]
    ptq_cpu = d["quantization_experiments"][0]["metrics"]["inference_cpu_ms"]
    best_cpu = d["etap7_experiments"]["best_optimized"]["metrics"]["inference_cpu_ms"]

    labels = (
        ["Baseline"] +
        [f"Struct. {int(e['sparsity']*100)}%" for e in struct_exps] +
        ["PTQ INT8", "Najlepszy\n+ Pruning + PTQ"]
    )
    times = (
        [baseline_cpu] +
        [e["metrics"]["inference_cpu_ms"] for e in struct_exps] +
        [ptq_cpu, best_cpu]
    )
    bar_colors = (
        [COLORS[0]] +
        [COLORS[1]] * len(struct_exps) +
        [COLORS[2], COLORS[3]]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, times, color=bar_colors, edgecolor="white")

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Czas wnioskowania CPU [ms/obraz]")
    ax.set_title("Czas wnioskowania na CPU – porównanie wariantów modelu")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("inference_time_comparison")


# ─── 5. Augmentation bar chart ────────────────────────────────────────────────

def plot_augmentation(d: dict):
    exps = d["etap7_experiments"]["augmentation"]
    names = [e["name"].replace(" (", "\n(") for e in exps]
    accs  = [e["metrics"]["test_acc"] * 100 for e in exps]
    f1s   = [e["metrics"]["test_f1"] * 100 for e in exps]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Dokładność", color=COLORS[0])
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1-score",   color=COLORS[1])

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Metryka [%]")
    ax.set_title("Etap 7 – wpływ augmentacji danych na jakość klasyfikacji")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(accs) - 5, max(accs) + 3)

    plt.tight_layout()
    save_fig("augmentation_comparison")


# ─── 6. Optimizer / scheduler comparison ─────────────────────────────────────

def plot_optimizer_scheduler(d: dict):
    e7 = d["etap7_experiments"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Optimizers
    opts = e7["optimizer"]
    o_names = [e["name"] for e in opts]
    o_accs  = [e["metrics"]["test_acc"] * 100 for e in opts]
    axes[0].bar(o_names, o_accs, color=COLORS[:len(opts)])
    for i, v in enumerate(o_accs):
        axes[0].text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Dokładność testowa [%]")
    axes[0].set_title("Porównanie optymalizatorów")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_ylim(min(o_accs) - 2, max(o_accs) + 2)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Schedulers
    scheds = e7["scheduler"]
    s_names = [e["name"] for e in scheds]
    s_accs  = [e["metrics"]["test_acc"] * 100 for e in scheds]
    axes[1].bar(s_names, s_accs, color=COLORS[:len(scheds)])
    for i, v in enumerate(s_accs):
        axes[1].text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Dokładność testowa [%]")
    axes[1].set_title("Porównanie harmonogramów LR")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylim(min(s_accs) - 2, max(s_accs) + 2)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha="right")

    fig.suptitle("Etap 7 – optymalizatory i harmonogramy szybkości uczenia", fontsize=13)
    plt.tight_layout()
    save_fig("optimizer_scheduler_comparison")


# ─── 7. Dropout regularization ────────────────────────────────────────────────

def plot_dropout(d: dict):
    exps = d["etap7_experiments"]["dropout"]
    probs = [e["dropout"] for e in exps]
    accs  = [e["metrics"]["test_acc"] * 100 for e in exps]
    f1s   = [e["metrics"]["test_f1"] * 100 for e in exps]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(probs, accs, "o-", color=COLORS[0], label="Dokładność", linewidth=2, markersize=7)
    ax.plot(probs, f1s,  "s--", color=COLORS[1], label="F1-score",  linewidth=2, markersize=7)
    ax.set_xlabel("Prawdopodobieństwo dropout p")
    ax.set_ylabel("Metryka [%]")
    ax.set_title("Etap 7 – wpływ współczynnika dropout na dokładność")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    save_fig("dropout_regularization")


# ─── 8. Final overview – accuracy vs size scatter ─────────────────────────────

def plot_accuracy_vs_size(d: dict):
    baseline  = d["baseline"]["metrics"]
    ptq       = d["quantization_experiments"][0]["metrics"]
    qat       = d["quantization_experiments"][1]["metrics"]
    pq50      = d["quantization_experiments"][2]["metrics"]
    pqs50     = d["quantization_experiments"][3]["metrics"]
    best_e7   = d["etap7_experiments"]["best_model"]["metrics"]
    best_opt  = d["etap7_experiments"]["best_optimized"]["metrics"]

    points = [
        (baseline["size_mb"],  baseline["test_acc"] * 100,  "Baseline",         "o", COLORS[0]),
        (ptq["size_mb"],       ptq["test_acc"] * 100,        "PTQ INT8",          "s", COLORS[1]),
        (qat["size_mb"],       qat["test_acc"] * 100,        "QAT INT8",          "^", COLORS[2]),
        (pq50["size_mb"],      pq50["test_acc"] * 100,       "Pruning 50%+PTQ",   "D", COLORS[3]),
        (pqs50["size_mb"],     pqs50["test_acc"] * 100,      "Struct.50%+PTQ",    "P", COLORS[4]),
        (best_e7["size_mb"],   best_e7["test_acc"] * 100,   "Najlepszy Etap~7",  "*", COLORS[6]),
        (best_opt["size_mb"],  best_opt["test_acc"] * 100,  "Najlepszy+Prune+PTQ","X",COLORS[7]),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for size, acc, label, marker, color in points:
        ax.scatter(size, acc, marker=marker, color=color, s=120, label=label, zorder=5)
        ax.annotate(label, (size, acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5)

    ax.set_xlabel("Rozmiar modelu [MB]")
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Kompromis dokładność–rozmiar modelu")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    save_fig("accuracy_vs_size")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating plots...")
    d = load_data()
    plot_training_curves(d)
    plot_pruning_tradeoff(d)
    plot_model_size(d)
    plot_inference_time(d)
    plot_augmentation(d)
    plot_optimizer_scheduler(d)
    plot_dropout(d)
    plot_accuracy_vs_size(d)
    print(f"\nAll figures written to {FIGURES_DIR}")
