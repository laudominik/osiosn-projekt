from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from generate_tables import load_data  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.bbox": "tight"
})
COLORS = plt.cm.tab10.colors


def save_fig(name: str):
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"  ✓  {name}.pdf")
    plt.close()


def _bl_pair(d, key, scale=1.0):
    bl_n = d.get("baseline", {}).get("noisy", {}) or {}
    bl_c = d.get("baseline", {}).get("clean", {}) or {}
    return (
        (bl_n.get(key) * scale if bl_n.get(key) is not None else None),
        (bl_c.get(key) * scale if bl_c.get(key) is not None else None),
    )


def plot_training_curves(d: dict):
    def _load_hist(name):
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                if "metrics" in data and "training_history" in data["metrics"]:
                    return data["metrics"]["training_history"]
        return None

    model_prefixes: set[str] = set()
    for file in RESULTS_DIR.glob("*.json"):
        name = file.stem
        if name.endswith("_noisy") or name.endswith("_clean"):
            base = name.replace("_noisy", "").replace("_clean", "")
            model_prefixes.add(base)

    for base_name in model_prefixes:
        hn = _load_hist(f"{base_name}_noisy")
        hc = _load_hist(f"{base_name}_clean")
        if not hn and not hc:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

        def _plot_curves(h, suffix, c_train, c_val, m_train, m_val):
            if not h:
                return
            epochs = h.get("epochs", list(range(1, len(h.get("train_loss", [])) + 1)))
            if "train_loss" in h and "val_loss" in h:
                ax1.plot(epochs, h["train_loss"], f"{m_train}-", color=c_train,
                         label=f"Trening {suffix}", markersize=4)
                ax1.plot(epochs, h["val_loss"], f"{m_val}--", color=c_val, alpha=0.8,
                         label=f"Walidacja {suffix}", markersize=4)
            if "train_acc" in h and "val_acc" in h:
                ax2.plot(epochs, [v * 100 for v in h["train_acc"]], f"{m_train}-",
                         color=c_train, label=f"Trening {suffix}", markersize=4)
                ax2.plot(epochs, [v * 100 for v in h["val_acc"]], f"{m_val}--",
                         color=c_val, alpha=0.8, label=f"Walidacja {suffix}", markersize=4)

        if hn: _plot_curves(hn, "(szum)",   COLORS[0], COLORS[1], "o", "s")
        if hc: _plot_curves(hc, "(czyste)", COLORS[2], COLORS[3], "^", "D")

        ax1.set_xlabel("Epoka"); ax1.set_ylabel("Loss")
        ax1.set_title("Krzywe strat"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Epoka"); ax2.set_ylabel("Dokładność [%]")
        ax2.set_title("Krzywe dokładności"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        fig.suptitle(f"Krzywe Uczenia: {base_name}", fontsize=13)
        plt.tight_layout()
        save_fig(f"training_curves_{base_name}")


def plot_pruning_tradeoff(d: dict):
    exps = d.get("pruning_experiments", [])
    bl_n, bl_c = _bl_pair(d, "test_acc", scale=100)

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = [
        ("unstruct", "o", "o-", COLORS[0], "UNSTRUCTURED ONESHOT"),
        ("unstruct", "s", "s-", COLORS[1], "UNSTRUCTURED SCHEDULED"),
        ("struct",   "o", "^-", COLORS[2], "STRUCTURED ONESHOT"),
        ("struct",   "s", "D-", COLORS[3], "STRUCTURED SCHEDULED"),
    ]

    for p_type, p_sched, style, color, label_pfx in variants:
        sub = [e for e in exps if e.get("type") == p_type and e.get("schedule") == p_sched]
        if not sub:
            continue
        sp_levels = sorted({e.get("sparsity", 0) * 100 for e in sub})

        sp_n, ac_n, sp_c, ac_c = [], [], [], []
        for s in sp_levels:
            match = next((e for e in sub if abs(e.get("sparsity", 0) * 100 - s) < 1.0), None)
            if not match:
                continue
            if match.get("noisy") and match["noisy"].get("test_acc"):
                sp_n.append(s); ac_n.append(match["noisy"]["test_acc"] * 100)
            if match.get("clean") and match["clean"].get("test_acc"):
                sp_c.append(s); ac_c.append(match["clean"]["test_acc"] * 100)

        if sp_n and bl_n is not None:
            ax.plot([0] + sp_n, [bl_n] + ac_n, style, color=color, label=f"{label_pfx} (Szum)")
        if sp_c and bl_c is not None:
            ax.plot([0] + sp_c, [bl_c] + ac_c, style.replace("-", "--"), color=color,
                    alpha=0.6, label=f"{label_pfx} (Czyste)")

    if bl_n is not None:
        ax.axhline(bl_n, color="gray", linestyle=":", linewidth=1.5, label="Baseline (Szum)")

    ax.set_xlabel("Sparsity [%]"); ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Wpływ przerzedzania na dokładność klasyfikacji")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    save_fig("pruning_accuracy_tradeoff")


def plot_pruning_inference_time(d: dict):
    exps = d.get("pruning_experiments", [])
    # Wyciągamy baseline dla czasu wnioskowania na CPU
    bl_n, bl_c = _bl_pair(d, "inference_cpu_ms", scale=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = [
        ("unstruct", "o", "o-", COLORS[0], "UNSTRUCTURED ONESHOT"),
        ("unstruct", "s", "s-", COLORS[1], "UNSTRUCTURED SCHEDULED"),
        ("struct",   "o", "^-", COLORS[2], "STRUCTURED ONESHOT"),
        ("struct",   "s", "D-", COLORS[3], "STRUCTURED SCHEDULED"),
    ]

    for p_type, p_sched, style, color, label_pfx in variants:
        sub = [e for e in exps if e.get("type") == p_type and e.get("schedule") == p_sched]
        if not sub:
            continue
        sp_levels = sorted({e.get("sparsity", 0) * 100 for e in sub})

        sp_n, inf_n, sp_c, inf_c = [], [], [], []
        for s in sp_levels:
            match = next((e for e in sub if abs(e.get("sparsity", 0) * 100 - s) < 1.0), None)
            if not match:
                continue
            if match.get("noisy") and match["noisy"].get("inference_cpu_ms"):
                sp_n.append(s); inf_n.append(match["noisy"]["inference_cpu_ms"])
            if match.get("clean") and match["clean"].get("inference_cpu_ms"):
                sp_c.append(s); inf_c.append(match["clean"]["inference_cpu_ms"])

        if sp_n and bl_n is not None:
            ax.plot([0] + sp_n, [bl_n] + inf_n, style, color=color, label=f"{label_pfx} (Szum)")
        if sp_c and bl_c is not None:
            ax.plot([0] + sp_c, [bl_c] + inf_c, style.replace("-", "--"), color=color,
                    alpha=0.6, label=f"{label_pfx} (Czyste)")

    if bl_n is not None:
        ax.axhline(bl_n, color="gray", linestyle=":", linewidth=1.5, label="Baseline (Szum)")

    ax.set_xlabel("Sparsity [%]")
    ax.set_ylabel("Czas wnioskowania CPU [ms / próbka]")
    ax.set_title("Wpływ przerzedzania na czas wnioskowania (Inference Time)")
    # Przenosimy legendę, bo dla inference time wykresy zazwyczaj idą w dół
    ax.legend(fontsize=8, loc="upper right") 
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    save_fig("pruning_inference_time_tradeoff")

def plot_inference_vs_trained_pruning(d: dict):
    """Compare accuracy: trained pruning vs inference-time vs ephemeral (noisy only)."""
    bl_n, _ = _bl_pair(d, "test_acc", scale=100)

    trained_exps = [e for e in d.get("pruning_experiments", [])
                    if e.get("type") == "unstruct" and e.get("schedule") == "o"]
    infer_exps  = d.get("pruning_infer", [])
    ephem_exps  = d.get("pruning_ephemeral", [])

    if not trained_exps and not infer_exps:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    def _extract(exps, tag):
        pts = sorted(
            [(e.get("sparsity", 0) * 100, e[tag]["test_acc"] * 100)
             for e in exps if e.get(tag) and e[tag].get("test_acc")],
            key=lambda x: x[0],
        )
        return [p[0] for p in pts], [p[1] for p in pts]

    sp_t, ac_t = _extract(trained_exps, "noisy")
    sp_i, ac_i = _extract(infer_exps, "noisy")
    sp_e, ac_e = _extract(ephem_exps, "noisy")

    if sp_t:
        xs = ([0] + sp_t) if bl_n is not None else sp_t
        ys = ([bl_n] + ac_t) if bl_n is not None else ac_t
        ax.plot(xs, ys, "o-", color=COLORS[0], label="Niestrukt. one-shot (z doszkalaniem)")
    if sp_i:
        xs = ([0] + sp_i) if bl_n is not None else sp_i
        ys = ([bl_n] + ac_i) if bl_n is not None else ac_i
        ax.plot(xs, ys, "s--", color=COLORS[1], label="Czas wnioskowania (bez treningu)")
    if sp_e:
        xs = ([0] + sp_e) if bl_n is not None else sp_e
        ys = ([bl_n] + ac_e) if bl_n is not None else ac_e
        ax.plot(xs, ys, "^:", color=COLORS[2], label="Efemeryczne (odwracalne)")

    if bl_n is not None:
        ax.axhline(bl_n, color="gray", linestyle=":", linewidth=1.5, label="Baseline")

    ax.set_xlabel("Sparsity [%]"); ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Przerzedzanie: trening vs czas wnioskowania vs efemeryczne (z szumem)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    save_fig("pruning_inference_vs_trained")


def plot_quantization_comparison(d: dict):
    """Bar chart: baseline vs PTQ vs QAT for both noise variants."""
    bl = d.get("baseline", {})
    quant_exps = d.get("quantization_experiments", [])

    if not quant_exps:
        return

    all_entries = [{"name": "Baseline", "noisy": bl.get("noisy"), "clean": bl.get("clean")}]
    all_entries += quant_exps

    labels, acc_n, acc_c = [], [], []
    for e in all_entries:
        m_n = e.get("noisy") or {}
        m_c = e.get("clean") or {}
        an  = m_n.get("test_acc")
        ac  = m_c.get("test_acc")
        if an is None and ac is None:
            continue
        labels.append(e["name"] if "name" in e else e.get("id", "?"))
        acc_n.append(an * 100 if an is not None else 0)
        acc_c.append(ac * 100 if ac is not None else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    bars_n = ax.bar(x - w / 2, acc_n, w, label="Z szumem",   color=COLORS[0], alpha=0.85)
    bars_c = ax.bar(x + w / 2, acc_c, w, label="Czyste dane", color=COLORS[1], alpha=0.85)

    for bar in bars_n + bars_c:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.replace("(INT8)", "\n(INT8)").replace("+", "\n+") for lbl in labels],
        fontsize=9,
    )
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Porównanie kwantyzacji: Baseline vs PTQ vs QAT")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ymin = max(0, min(v for v in acc_n + acc_c if v > 0) - 5)
    ax.set_ylim(ymin, 100)
    plt.tight_layout()
    save_fig("quantization_comparison")


def plot_hyperopt_group(rows: list[dict], fig_name: str, title: str):
    """Grouped bar chart comparing noisy vs clean accuracy for a hyperopt group."""
    labels, acc_n, acc_c = [], [], []
    for e in rows:
        mn = e.get("noisy") or {}
        mc = e.get("clean") or {}
        an = mn.get("test_acc")
        ac = mc.get("test_acc")
        if an is None and ac is None:
            continue
        labels.append(e.get("name", e.get("id", "?")))
        acc_n.append(an * 100 if an is not None else 0)
        acc_c.append(ac * 100 if ac is not None else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.6), 4.5))
    bars_n = ax.bar(x - w / 2, acc_n, w, label="Z szumem",    color=COLORS[0], alpha=0.85)
    bars_c = ax.bar(x + w / 2, acc_c, w, label="Czyste dane", color=COLORS[1], alpha=0.85)

    for bar in bars_n + bars_c:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ymin = max(0, min(v for v in acc_n + acc_c if v > 0) - 5) if any(v > 0 for v in acc_n + acc_c) else 0
    ax.set_ylim(ymin, 100)
    plt.tight_layout()
    save_fig(fig_name)


def plot_hyperopt(d: dict):
    e7 = d.get("etap7_experiments", {})
    plot_hyperopt_group(
        e7.get("augmentation", []),
        "hyperopt_augmentation",
        "Wpływ strategii augmentacji danych",
    )
    plot_hyperopt_group(
        e7.get("optimizer", []),
        "hyperopt_optimizer",
        "Porównanie optymalizatorów",
    )
    plot_hyperopt_group(
        e7.get("scheduler", []),
        "hyperopt_scheduler",
        "Porównanie harmonogramów uczenia",
    )
    plot_hyperopt_group(
        e7.get("dropout", []),
        "hyperopt_dropout",
        "Wpływ współczynnika dropout",
    )


if __name__ == "__main__":
    print("Generating plots...")
    d = load_data()
    plot_training_curves(d)
    plot_pruning_tradeoff(d)
    plot_pruning_inference_time(d)
    plot_inference_vs_trained_pruning(d)
    plot_quantization_comparison(d)
    plot_hyperopt(d)
    print(f"\nAll figures written to {FIGURES_DIR}")
