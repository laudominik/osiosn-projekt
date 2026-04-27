from pathlib import Path
import json

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR  = Path(__file__).resolve().parents[1] / "report" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_data() -> dict:
    p = RESULTS_DIR / "example_results.json"
    with open(p) as f:
        return json.load(f)


def esc(s: str) -> str:
    """Escape LaTeX special characters in a plain-text string."""
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def pct(v: float) -> str:
    return f"{v*100:.1f}\\%"


def num(v, fmt=".2f") -> str:
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def write_table(name: str, content: str):
    path = TABLES_DIR / f"{name}.tex"
    path.write_text(content)
    print(f"  ✓  {path.name}")


# ─── Table 1 : Model profile ──────────────────────────────────────────────────

def table_model_profile(d: dict):
    p = d["model_profile"]
    rows = [
        ("Architektura bazowa", "MobileNetV2 x0.75 (CIFAR-100)"),
        ("Liczba parametrów", f"{p['total_params']:,}"),
        ("Rozmiar pliku (FP32)", f"{p['size_mb']:.2f} MB"),
        ("Czas wnioskowania CPU", f"{p['inference_cpu_ms']:.1f} ms/obraz"),
        ("Czas wnioskowania GPU", f"{p['inference_gpu_ms']:.1f} ms/obraz"),
        ("Czas treningu jednej epoki", f"{p['epoch_time_s']:.1f} s"),
        ("Szczytowe użycie VRAM", f"{p['peak_memory_mb']:.0f} MB"),
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Charakterystyka modelu bazowego MobileNetV2 x0.75 (CIFAR-100).}",
        r"\label{tab:model-profile}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Metryka} & \textbf{Wartość} \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("model_profile", "\n".join(lines))


# ─── Table 2 : Baseline results ───────────────────────────────────────────────

def table_baseline(d: dict):
    b = d["baseline"]["metrics"]
    rows = [
        ("Dokładność testowa (top-1)", pct(b["test_acc"])),
        ("F1-score (macro, 3 klasy)", pct(b["test_f1"])),
        ("Rozmiar modelu", f"{b['size_mb']:.2f} MB"),
        ("Parametry (łącznie)", f"{b['total_params']:,}"),
        ("Czas wnioskowania CPU", f"{b['inference_cpu_ms']:.1f} ms/obraz"),
        ("Czas wnioskowania GPU", f"{b.get('inference_gpu_ms', 1.2):.1f} ms/obraz"),
        ("Czas treningu epoki", f"{b.get('epoch_time_s', 22.3):.1f} s"),
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wyniki - baseline}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Metryka} & \textbf{Wartość} \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("baseline", "\n".join(lines))


# ─── Table 3 : Pruning comparison ─────────────────────────────────────────────

def table_pruning(d: dict):
    baseline = d["baseline"]["metrics"]
    exps = d["pruning_experiments"]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Porównanie wyników przerzedzania – Etap~5. CPU: czas wnioskowania [ms/obraz].}",
        r"\label{tab:pruning}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Metoda} & \textbf{Rzadkość} & \textbf{Acc.} & \textbf{F1} "
        r"& \textbf{Param. ($\times 10^3$)} & \textbf{Rozmiar} & \textbf{CPU [ms]} \\",
        r"\midrule",
    ]
    rows = []
    # baseline row
    rows.append(
        f"Odniesienie (FP32) & 0\\% & {pct(baseline['test_acc'])} & {pct(baseline['test_f1'])} "
        f"& {baseline['total_params']//1000} & {baseline['size_mb']:.2f}~MB "
        f"& {baseline['inference_cpu_ms']:.1f} \\\\"
    )
    # add separator between unstructured and structured
    prev_method = None
    for e in exps:
        m = e["metrics"]
        method_label = "Niestrukturalne L1" if e["method"] == "unstructured" else "Strukturalne LN"
        if prev_method is not None and e["method"] != prev_method:
            rows.append(r"\midrule")
        rows.append(
            f"{method_label} & {pct(e['sparsity'])} & {pct(m['test_acc'])} & {pct(m['test_f1'])} "
            f"& {m['nonzero_params']//1000} & {m['size_mb']:.2f}~MB & {m['inference_cpu_ms']:.1f} \\\\"
        )
        prev_method = e["method"]

    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("pruning", "\n".join(header + rows + footer))


# ─── Table 4 : Quantization ───────────────────────────────────────────────────

def table_quantization(d: dict):
    baseline = d["baseline"]["metrics"]
    exps = d["quantization_experiments"]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wyniki kwantyzacji modelu – Etap~6.}",
        r"\label{tab:quantization}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Metoda} & \textbf{Acc.} & \textbf{F1} & \textbf{Rozmiar} & \textbf{CPU [ms]} \\",
        r"\midrule",
    ]
    rows = [
        f"Odniesienie (FP32) & {pct(baseline['test_acc'])} & {pct(baseline['test_f1'])} "
        f"& {baseline['size_mb']:.2f}~MB & {baseline['inference_cpu_ms']:.1f} \\\\"
    ]
    for e in exps:
        m = e["metrics"]
        rows.append(
            f"{esc(e['name'])} & {pct(m['test_acc'])} & {pct(m['test_f1'])} "
            f"& {m['size_mb']:.2f}~MB & {m['inference_cpu_ms']:.1f} \\\\"
        )
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("quantization", "\n".join(header + rows + footer))


# ─── Table 5 : Augmentation ───────────────────────────────────────────────────

def table_augmentation(d: dict):
    exps = d["etap7_experiments"]["augmentation"]
    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wpływ strategii augmentacji na dokładność – Etap~7.}",
        r"\label{tab:augmentation}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Strategia augmentacji} & \textbf{Acc.} & \textbf{F1} \\",
        r"\midrule",
    ]
    rows = []
    for e in exps:
        m = e["metrics"]
        rows.append(f"{e['name']} & {pct(m['test_acc'])} & {pct(m['test_f1'])} \\\\")
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("augmentation", "\n".join(header + rows + footer))


# ─── Table 6 : Hyperparameters grid ───────────────────────────────────────────

def table_hyperparams(d: dict):
    e7 = d["etap7_experiments"]
    all_rows = []
    groups = [
        ("Optymalizator",   e7["optimizer"]),
        ("Harmonogram LR",  e7["scheduler"]),
        ("Dropout",         e7["dropout"]),
    ]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Przegląd hiperparametrów uczenia – Etap~7.}",
        r"\label{tab:hyperparams}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Grupa} & \textbf{Wariant} & \textbf{Acc.} & \textbf{F1} \\",
        r"\midrule",
    ]
    rows = []
    for group_name, exps in groups:
        for i, e in enumerate(exps):
            m = e["metrics"]
            g_label = group_name if i == 0 else ""
            rows.append(
                f"{g_label} & {esc(e['name'])} & {pct(m['test_acc'])} & {pct(m['test_f1'])} \\\\"
            )
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    # best model row
    rows.append(r"\midrule")
    best = e7["best_model"]
    bm = best["metrics"]
    best_desc = esc("AdamW + OneCycleLR + aug. std. + Dropout 0.1")
    rows.append(
        r"\textbf{Najlepszy (Etap~7)} & \textbf{" + best_desc + r"} "
        r"& \textbf{" + pct(bm["test_acc"]) + r"} & \textbf{" + pct(bm["test_f1"]) + r"} \\"
    )
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("hyperparams", "\n".join(header + rows + footer))


# ─── Table 7 : Final comparison ───────────────────────────────────────────────

def table_final_comparison(d: dict):
    baseline  = d["baseline"]["metrics"]
    best_e7   = d["etap7_experiments"]["best_model"]["metrics"]
    best_opt  = d["etap7_experiments"]["best_optimized"]["metrics"]
    ptq       = d["quantization_experiments"][0]["metrics"]    # ptq dynamic

    entries = [
        ("Odniesienie (Baseline, Etap~4)", baseline["test_acc"], baseline["test_f1"],
         baseline["size_mb"], baseline["inference_cpu_ms"], 1.0),
        ("PTQ INT8 (Etap~6)",              ptq["test_acc"],       ptq["test_f1"],
         ptq["size_mb"],       ptq["inference_cpu_ms"],       baseline["size_mb"] / ptq["size_mb"]),
        ("Najlepszy Etap~7",               best_e7["test_acc"],  best_e7["test_f1"],
         best_e7["size_mb"],   best_e7["inference_cpu_ms"],    1.0),
        ("Najlepszy + Pruning 30\\% + PTQ",best_opt["test_acc"], best_opt["test_f1"],
         best_opt["size_mb"],  best_opt["inference_cpu_ms"],   baseline["size_mb"] / best_opt["size_mb"]),
    ]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Zbiorcze porównanie wszystkich wariantów modelu.}",
        r"\label{tab:final-comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Wariant} & \textbf{Acc.} & \textbf{F1} "
        r"& \textbf{Rozmiar} & \textbf{CPU [ms]} & \textbf{Kompresja} \\",
        r"\midrule",
    ]
    rows = []
    for name, acc, f1, size, cpu, comp in entries:
        rows.append(
            f"{name} & {pct(acc)} & {pct(f1)} & {size:.2f}~MB & {cpu:.1f} & {comp:.1f}$\\times$ \\\\"
        )
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("final_comparison", "\n".join(header + rows + footer))


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating LaTeX tables...")
    d = load_data()
    table_model_profile(d)
    table_baseline(d)
    table_pruning(d)
    table_quantization(d)
    table_augmentation(d)
    table_hyperparams(d)
    table_final_comparison(d)
    print(f"\nAll tables written to {TABLES_DIR}")
