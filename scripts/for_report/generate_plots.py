"""
Generuje wykresy do raportu LaTeX z wynikami eksperymentów.

Czyta: wyniki z katalogu results/ lub przykładowe z example_results.json
Zapisuje: report/figures/*.pdf  (i *.png do podglądu)

Uruchomienie:
    uv run python scripts/generate_plots.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from generate_tables import load_data  # noqa: E402

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


def _m_pair(entry, key, scale=1.0):
    """Return (noisy_val, clean_val) for an entry, or (val, None) in old format."""
    if "metrics" in entry:
        v = entry["metrics"].get(key)
        return (v * scale if v is not None else None), None
    vn = (entry.get("noisy") or {}).get(key)
    vc = (entry.get("clean") or {}).get(key)
    return (vn * scale if vn is not None else None), (vc * scale if vc is not None else None)


def _bl_pair(d, key, scale=1.0):
    bl = d["baseline"]
    if "metrics" in bl:
        v = bl["metrics"].get(key)
        return (v * scale if v is not None else None), None
    vn = (bl.get("noisy") or {}).get(key)
    vc = (bl.get("clean") or {}).get(key)
    return (vn * scale if vn is not None else None), (vc * scale if vc is not None else None)



def plot_training_curves(d: dict):
    def _load_hist(name):
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                    # Szukamy historii na wierzchu albo w słowniku "metrics"
                    if "training_history" in data:
                        return data["training_history"]
                    if "metrics" in data and "training_history" in data["metrics"]:
                        return data["metrics"]["training_history"]
            except Exception as e:
                print(f"  ⚠  Błąd czytania pliku {p}: {e}")
        return None

    models_to_plot = [
        ("baseline", "Baseline - finetuning"),
        # ("best_etap7", "Etap 7 – najlepszy zoptymalizowany model"),
        # ("prune_struct_50", "Etap 5 – Pruning strukturalny 50%"),
        # ("prune_unstruct_50", "Etap 5 – Pruning niestrukturalny 50%"),
    ]

    for base_name, title in models_to_plot:
        hn = _load_hist(f"{base_name}_noisy")
        hc = _load_hist(f"{base_name}_clean")

        if not hn and not hc and d.get("_source") == "example":
            if base_name in d and "training_history" in d[base_name]:
                hn = d[base_name]["training_history"]
            elif base_name == "baseline" and "training_history" in d.get("baseline", {}):
                hn = d["baseline"]["training_history"]

        if not hn and not hc:
            print(f"  ⚠  Brak danych training_history dla {base_name}_noisy/clean. Pomijam rysowanie krzywych dla tego modelu.")
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

        def _plot_curves(h, suffix, c_train, c_val, m_train, m_val):
            if not h: return
            
            epochs = h.get("epochs", list(range(1, len(h.get("train_loss", [])) + 1)))
            
            if "train_loss" in h and "val_loss" in h and len(h["train_loss"]) > 0:
                ax1.plot(epochs, h["train_loss"], f"{m_train}-", color=c_train, label=f"Trening {suffix}")
                ax1.plot(epochs, h["val_loss"],   f"{m_val}--", color=c_val, alpha=0.8, label=f"Walidacja {suffix}")

            if "train_acc" in h and "val_acc" in h and len(h["train_acc"]) > 0:
                ax2.plot(epochs, [v * 100 for v in h["train_acc"]], f"{m_train}-",  color=c_train, label=f"Trening {suffix}")
                ax2.plot(epochs, [v * 100 for v in h["val_acc"]],   f"{m_val}--", color=c_val, alpha=0.8, label=f"Walidacja {suffix}")

        if hn:
            _plot_curves(hn, "(z szumem)", COLORS[0], COLORS[1], "o", "s")
            
        if hc:
            _plot_curves(hc, "(czyste)", COLORS[2], COLORS[3], "^", "D")

        ax1.set_xlabel("Epoka")
        ax1.set_ylabel("Loss (CrossEntropy)")
        ax1.set_title("Krzywe strat")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Dokładność [%]")
        ax2.set_title("Krzywe dokładności")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        
        save_fig(f"training_curves_{base_name}")


def plot_pruning_tradeoff(d: dict):
    exps   = d["pruning_experiments"]
    bl_n, bl_c = _bl_pair(d, "test_acc", scale=100)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, style, color, label_pfx in [
        ("unstructured", "o-",  COLORS[0], "Niestrukturalne L1"),
        ("structured",   "s--", COLORS[1], "Strukturalne LN"),
    ]:
        sub = [e for e in exps if e["method"] == method]
        if not sub:
            continue
        sp  = [e["sparsity"] * 100 for e in sub]
        sp  = sorted(set(sp))

        # noisy series
        sp_n = sorted({e["sparsity"] * 100 for e in sub if _m_pair(e, "test_acc")[0] is not None})
        ac_n = []
        for s in sp_n:
            vals = [_m_pair(e, "test_acc", 100)[0] for e in sub if e["sparsity"]*100 == s and _m_pair(e, "test_acc")[0] is not None]
            ac_n.append(vals[0] if vals else None)
        if bl_n is not None and sp_n:
            ax.plot([0] + sp_n, [bl_n] + ac_n, style, color=color,
                    label=f"{label_pfx} (z szumem)")

        # clean series
        sp_c = sorted({e["sparsity"] * 100 for e in sub if _m_pair(e, "test_acc")[1] is not None})
        ac_c = []
        for s in sp_c:
            vals = [_m_pair(e, "test_acc", 100)[1] for e in sub if e["sparsity"]*100 == s and _m_pair(e, "test_acc")[1] is not None]
            ac_c.append(vals[0] if vals else None)
        if bl_c is not None and sp_c:
            ax.plot([0] + sp_c, [bl_c] + ac_c,
                    style.replace("o", "^").replace("s", "D"),
                    color=color, alpha=0.6, linestyle=":",
                    label=f"{label_pfx} (czyste)")

    bl_plot = bl_n if bl_n is not None else bl_c
    if bl_plot:
        ax.axhline(bl_plot, color="gray", linestyle=":", linewidth=1, label="Baseline (brak przerzedzania)")

    ax.set_xlabel("Rzadkość [%]")
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Etap 5 – wpływ przerzedzania na dokładność klasyfikacji")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    save_fig("pruning_accuracy_tradeoff")


def plot_model_size(d: dict):
    source = d.get("_source", "example")

    if source == "example":
        baseline_size = d["baseline"]["metrics"]["size_mb"]
        quant = d["quantization_experiments"]
        ptq  = quant[0]["metrics"]
        qat  = quant[1]["metrics"] if len(quant) > 1 else {}
        pq   = quant[2]["metrics"] if len(quant) > 2 else {}
        best = d["etap7_experiments"]["best_optimized"]["metrics"]
        labels = ["Baseline\n(FP32)", "PTQ INT8\n(Etap 6)", "QAT INT8\n(Etap 6)",
                  "Pruning 50%\n+ PTQ", "Najlepszy\n+ Pruning 30%\n+ PTQ"]
        sizes = [baseline_size, ptq.get("size_mb"), qat.get("size_mb"),
                 pq.get("size_mb"), best.get("size_mb")]
        colors_bar = [COLORS[0], COLORS[1], COLORS[1], COLORS[2], COLORS[3]]
    else:
        bl_n = (d["baseline"].get("noisy") or {}).get("size_mb")
        bl_c = (d["baseline"].get("clean") or {}).get("size_mb")
        ptq_n = next(((e.get("noisy") or {}).get("size_mb") for e in d["quantization_experiments"]
                      if e["id"] == "quant_ptq_dynamic"), None)
        ptq_c = next(((e.get("clean") or {}).get("size_mb") for e in d["quantization_experiments"]
                      if e["id"] == "quant_ptq_dynamic"), None)
        pq_n = next(((e.get("noisy") or {}).get("size_mb") for e in d["quantization_experiments"]
                     if e["id"] == "prune50_quant"), None)
        pq_c = next(((e.get("clean") or {}).get("size_mb") for e in d["quantization_experiments"]
                     if e["id"] == "prune50_quant"), None)
        labels = ["Baseline\nZ szumem", "Baseline\nCzyste",
                  "PTQ\nZ szumem", "PTQ\nCzyste",
                  "Prune50%+PTQ\nZ szumem", "Prune50%+PTQ\nCzyste"]
        sizes = [bl_n, bl_c, ptq_n, ptq_c, pq_n, pq_c]
        colors_bar = [COLORS[0], COLORS[0], COLORS[1], COLORS[1], COLORS[2], COLORS[2]]

    valid = [(l, s, c) for l, s, c in zip(labels, sizes, colors_bar) if s is not None]
    if not valid:
        return
    labels, sizes, colors_bar = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(len(labels) * 1.5, 8), 5))
    bars = ax.bar(labels, sizes, color=colors_bar, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f} MB", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Rozmiar modelu [MB]")
    ax.set_title("Porównanie rozmiarów modelu po różnych optymalizacjach")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(sizes) * 1.25)
    
    # Zapobieganie zlewaniu się tekstów
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()
    save_fig("model_size_comparison")


# ─── 4. Inference time comparison ────────────────────────────────────────────

def plot_inference_time(d: dict):
    source = d.get("_source", "example")

    if source == "example":
        baseline_cpu = d["baseline"]["metrics"]["inference_cpu_ms"]
        struct_exps = [e for e in d["pruning_experiments"] if e["method"] == "structured"]
        ptq_cpu  = d["quantization_experiments"][0]["metrics"]["inference_cpu_ms"]
        best_cpu = d["etap7_experiments"]["best_optimized"]["metrics"]["inference_cpu_ms"]

        labels = (["Baseline"] +
                  [f"Struct. {int(e['sparsity']*100)}%" for e in struct_exps] +
                  ["PTQ INT8", "Najlepszy\n+ Pruning + PTQ"])
        times  = ([baseline_cpu] +
                  [e["metrics"]["inference_cpu_ms"] for e in struct_exps] +
                  [ptq_cpu, best_cpu])
        bar_colors = ([COLORS[0]] + [COLORS[1]] * len(struct_exps) + [COLORS[2], COLORS[3]])
    else:
        bl_n = (d["baseline"].get("noisy") or {}).get("inference_cpu_ms")
        bl_c = (d["baseline"].get("clean") or {}).get("inference_cpu_ms")
        struct_exps = [e for e in d["pruning_experiments"] if e["method"] == "structured"]
        ptq_n = next(((e.get("noisy") or {}).get("inference_cpu_ms")
                      for e in d["quantization_experiments"] if e["id"] == "quant_ptq_dynamic"), None)
        ptq_c = next(((e.get("clean") or {}).get("inference_cpu_ms")
                      for e in d["quantization_experiments"] if e["id"] == "quant_ptq_dynamic"), None)
        entries = []
        if bl_n: entries.append(("Baseline\nZ szumem", bl_n, COLORS[0]))
        if bl_c: entries.append(("Baseline\nCzyste",   bl_c, COLORS[0]))
        for e in struct_exps:
            n = (e.get("noisy") or {}).get("inference_cpu_ms")
            c = (e.get("clean") or {}).get("inference_cpu_ms")
            sp = int(e["sparsity"]*100)
            if n: entries.append((f"Struct.{sp}%\nZ szumem", n, COLORS[1]))
            if c: entries.append((f"Struct.{sp}%\nCzyste",   c, COLORS[1]))
        if ptq_n: entries.append(("PTQ\nZ szumem", ptq_n, COLORS[2]))
        if ptq_c: entries.append(("PTQ\nCzyste",   ptq_c, COLORS[2]))
        if not entries:
            return
        labels, times, bar_colors = zip(*entries)

    fig, ax = plt.subplots(figsize=(max(len(labels) * 1.0, 8), 5))
    bars = ax.bar(labels, times, color=bar_colors, edgecolor="white")

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Czas wnioskowania CPU [ms/obraz]")
    ax.set_title("Czas wnioskowania na CPU – porównanie wariantów modelu")
    ax.grid(True, axis="y", alpha=0.3)

    # Zapobieganie zlewaniu się tekstów przy dużej ilości wariantów
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()
    save_fig("inference_time_comparison")


# ─── 5. Augmentation bar chart ────────────────────────────────────────────────

def plot_augmentation(d: dict):
    source = d.get("_source", "example")
    exps = d["etap7_experiments"]["augmentation"]
    names = [e["name"].replace(" (", "\n(") for e in exps]
    x = np.arange(len(names))

    if source == "example":
        accs = [e["metrics"]["test_acc"] * 100 for e in exps]
        f1s  = [e["metrics"]["test_f1"]  * 100 for e in exps]
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - w/2, accs, w, label="Dokładność", color=COLORS[0])
        bars2 = ax.bar(x + w/2, f1s,  w, label="F1-score",   color=COLORS[1])
        for b in list(bars1) + list(bars2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Metryka [%]")
        ax.set_title("Etap 7 – wpływ augmentacji danych na jakość klasyfikacji")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(min(accs + f1s) - 5, max(accs + f1s) + 3)
    else:
        accs_n = [((e.get("noisy") or {}).get("test_acc", 0) or 0) * 100 for e in exps]
        accs_c = [((e.get("clean") or {}).get("test_acc", 0) or 0) * 100 for e in exps]
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, accs_n, w, label="Z szumem", color=COLORS[0])
        ax.bar(x + w/2, accs_c, w, label="Czyste",   color=COLORS[2])
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Dokładność testowa [%]")
        ax.set_title("Etap 7 – wpływ augmentacji danych (z szumem vs czyste)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("augmentation_comparison")


# ─── 6. Optimizer / scheduler comparison ─────────────────────────────────────

def plot_optimizer_scheduler(d: dict):
    source = d.get("_source", "example")
    e7 = d["etap7_experiments"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _bar_group(ax, exps, title):
        names = [e["name"] for e in exps]
        x = np.arange(len(names))
        if source == "example":
            accs = [e["metrics"]["test_acc"] * 100 for e in exps]
            ax.bar(x, accs, color=COLORS[:len(exps)])
            for i, v in enumerate(accs):
                ax.text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(min(accs) - 2, max(accs) + 2)
        else:
            accs_n = [((e.get("noisy") or {}).get("test_acc") or 0) * 100 for e in exps]
            accs_c = [((e.get("clean") or {}).get("test_acc") or 0) * 100 for e in exps]
            w = 0.35
            ax.bar(x - w/2, accs_n, w, label="Z szumem", color=COLORS[0])
            ax.bar(x + w/2, accs_c, w, label="Czyste",   color=COLORS[2])
            ax.legend(fontsize=9)
            all_v = [v for v in accs_n + accs_c if v]
            if all_v:
                ax.set_ylim(min(all_v) - 2, max(all_v) + 2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Dokładność testowa [%]")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    _bar_group(axes[0], e7["optimizer"], "Porównanie optymalizatorów")
    _bar_group(axes[1], e7["scheduler"], "Porównanie harmonogramów LR")

    fig.suptitle("Etap 7 – optymalizatory i harmonogramy szybkości uczenia", fontsize=13)
    plt.tight_layout()
    save_fig("optimizer_scheduler_comparison")


# ─── 7. Dropout regularization ────────────────────────────────────────────────

def plot_dropout(d: dict):
    source = d.get("_source", "example")
    exps = d["etap7_experiments"]["dropout"]

    # dropout probability is encoded in the experiment id: dropout_XX
    probs = []
    for e in exps:
        if "dropout" in e:
            probs.append(float(e["dropout"]))
        else:
            raw = e["id"].split("_")[-1]
            probs.append(int(raw) / 10)

    fig, ax = plt.subplots(figsize=(7, 5))

    if source == "example":
        accs = [e["metrics"]["test_acc"] * 100 for e in exps]
        f1s  = [e["metrics"]["test_f1"]  * 100 for e in exps]
        ax.plot(probs, accs, "o-",  color=COLORS[0], label="Dokładność", linewidth=2, markersize=7)
        ax.plot(probs, f1s,  "s--", color=COLORS[1], label="F1-score",   linewidth=2, markersize=7)
    else:
        accs_n = [((e.get("noisy") or {}).get("test_acc") or 0) * 100 for e in exps]
        accs_c = [((e.get("clean") or {}).get("test_acc") or 0) * 100 for e in exps]
        ax.plot(probs, accs_n, "o-",  color=COLORS[0], label="Z szumem",  linewidth=2, markersize=7)
        ax.plot(probs, accs_c, "s--", color=COLORS[2], label="Czyste",    linewidth=2, markersize=7)

    ax.set_xlabel("Prawdopodobieństwo dropout p")
    ax.set_ylabel("Dokładność testowa [%]")
    ax.set_title("Etap 7 – wpływ współczynnika dropout na dokładność")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    save_fig("dropout_regularization")


# ─── 8. Final overview – accuracy vs size scatter ─────────────────────────────

def plot_accuracy_vs_size(d: dict):
    source = d.get("_source", "example")

    if source == "example":
        baseline  = d["baseline"]["metrics"]
        ptq       = d["quantization_experiments"][0]["metrics"]
        qat       = d["quantization_experiments"][1]["metrics"] if len(d["quantization_experiments"]) > 1 else {}
        pq50      = d["quantization_experiments"][2]["metrics"] if len(d["quantization_experiments"]) > 2 else {}
        best_e7   = d["etap7_experiments"]["best_model"]["metrics"]
        best_opt  = d["etap7_experiments"]["best_optimized"]["metrics"]

        points = [
            (baseline.get("size_mb"),  baseline.get("test_acc", 0) * 100, "Baseline",         "o", COLORS[0]),
            (ptq.get("size_mb"),       ptq.get("test_acc", 0) * 100,       "PTQ INT8",          "s", COLORS[1]),
            (qat.get("size_mb"),       qat.get("test_acc", 0) * 100,       "QAT INT8",          "^", COLORS[2]),
            (pq50.get("size_mb"),      pq50.get("test_acc", 0) * 100,      "Pruning 50%+PTQ",   "D", COLORS[3]),
            (best_e7.get("size_mb"),   best_e7.get("test_acc", 0) * 100,  "Najlepszy Etap~7",  "*", COLORS[6]),
            (best_opt.get("size_mb"),  best_opt.get("test_acc", 0) * 100, "Najlepszy+Prune+PTQ","X", COLORS[7]),
        ]
    else:
        def _extract(d_src, key_id, tag):
            if key_id == "baseline":
                m = (d_src["baseline"].get(tag) or {})
            else:
                m = next(((e.get(tag) or {}) for e in d_src.get("quantization_experiments", [])
                          if e["id"] == key_id), {})
            return m.get("size_mb"), (m.get("test_acc") or 0) * 100

        bl_n  = _extract(d, "baseline",         "noisy")
        bl_c  = _extract(d, "baseline",         "clean")
        ptq_n = _extract(d, "quant_ptq_dynamic", "noisy")
        ptq_c = _extract(d, "quant_ptq_dynamic", "clean")
        pq_n  = _extract(d, "prune50_quant",    "noisy")
        pq_c  = _extract(d, "prune50_quant",    "clean")
        best_n = ((d["etap7_experiments"]["best_model"].get("noisy") or {}).get("size_mb"),
                  ((d["etap7_experiments"]["best_model"].get("noisy") or {}).get("test_acc") or 0) * 100)
        best_c = ((d["etap7_experiments"]["best_model"].get("clean") or {}).get("size_mb"),
                  ((d["etap7_experiments"]["best_model"].get("clean") or {}).get("test_acc") or 0) * 100)
        points = [
            (bl_n[0],   bl_n[1],   "Baseline (sz.)",   "o",  COLORS[0]),
            (bl_c[0],   bl_c[1],   "Baseline (cz.)",   "o",  COLORS[5]),
            (ptq_n[0],  ptq_n[1],  "PTQ (sz.)",         "s",  COLORS[1]),
            (ptq_c[0],  ptq_c[1],  "PTQ (cz.)",         "s",  COLORS[6]),
            (pq_n[0],   pq_n[1],   "Prune+PTQ (sz.)",  "D",  COLORS[2]),
            (pq_c[0],   pq_c[1],   "Prune+PTQ (cz.)",  "D",  COLORS[7]),
            (best_n[0], best_n[1], "Najlepszy (sz.)",   "*",  COLORS[3]),
            (best_c[0], best_c[1], "Najlepszy (cz.)",   "*",  COLORS[4]),
        ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for size, acc, label, marker, color in points:
        if not size or not acc:
            continue
        ax.scatter(size, acc, marker=marker, color=color, s=120, label=label, zorder=5)
        ax.annotate(label, (size, acc), textcoords="offset points", xytext=(6, 4), fontsize=8.5)

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
    src = d.get("_source", "?")
    print(f"  Data source: {src}")
    plot_training_curves(d)
    # plot_pruning_tradeoff(d)
    # plot_model_size(d)
    # plot_inference_time(d)
    # plot_augmentation(d)
    # plot_optimizer_scheduler(d)
    # plot_dropout(d)
    # plot_accuracy_vs_size(d)
    # print(f"\nAll figures written to {FIGURES_DIR}")