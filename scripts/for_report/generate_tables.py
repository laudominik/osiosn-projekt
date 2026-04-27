from pathlib import Path
import json

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR  = Path(__file__).resolve().parents[1] / "report" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v*100:.1f}\\%"


def num(v, fmt=".2f") -> str:
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def write_table(name: str, content: str):
    path = TABLES_DIR / f"{name}.tex"
    path.write_text(content)
    print(f"  ✓  {path.name}")


def try_load(exp_id: str) -> dict | None:
    path = RESULTS_DIR / f"{exp_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_example() -> dict:
    p = RESULTS_DIR / "example_results.json"
    with open(p) as f:
        return json.load(f)


def load_data() -> dict:
    baseline_noisy = try_load("baseline_noisy")
    baseline_clean = try_load("baseline_clean")

    if baseline_noisy is None and baseline_clean is None:
        # No actual results – use example data (treat as noisy-only)
        ex = load_example()
        ex["_source"] = "example"
        return ex

    # Build normalised structure from individual files
    d: dict = {"_source": "actual"}

    # baseline
    d["baseline"] = {
        "noisy": baseline_noisy["metrics"] if baseline_noisy else None,
        "clean": baseline_clean["metrics"] if baseline_clean else None,
    }

    # model_profile from whichever baseline has it
    for tag in ("noisy", "clean"):
        b = try_load(f"baseline_{tag}")
        if b:
            mp = {k: b["metrics"].get(k) for k in
                  ("total_params", "size_mb", "inference_cpu_ms",
                   "inference_gpu_ms", "epoch_time_s", "peak_memory_mb")}
            d["model_profile"] = mp
            break

    # pruning experiments
    prune_exps = []
    for method, pfx, levels in [
        ("unstructured", "prune_unstruct", [10, 30, 50, 70, 90]),
        ("structured",   "prune_struct",   [10, 30, 50, 70]),
    ]:
        for sp in levels:
            entry = {"method": method, "sparsity": sp / 100, "noisy": None, "clean": None}
            for tag in ("noisy", "clean"):
                r = try_load(f"{pfx}_{sp:02d}_{tag}")
                if r:
                    entry[tag] = r["metrics"]
            if entry["noisy"] or entry["clean"]:
                prune_exps.append(entry)
    d["pruning_experiments"] = prune_exps

    # quantization experiments
    quant_exps = []
    for qid, qname in [
        ("quant_ptq_dynamic", "PTQ dynamiczna (INT8)"),
        ("prune50_quant",     "Pruning 50\\% + PTQ INT8"),
    ]:
        entry = {"id": qid, "name": qname, "noisy": None, "clean": None}
        for tag in ("noisy", "clean"):
            r = try_load(f"{qid}_{tag}")
            if r:
                entry[tag] = r["metrics"]
        if entry["noisy"] or entry["clean"]:
            quant_exps.append(entry)
    d["quantization_experiments"] = quant_exps

    # etap7 experiments
    aug_map = [
        ("aug_none",       "Brak augmentacji"),
        ("aug_basic",      "Podstawowa (crop+flip)"),
        ("aug_standard",   "Standardowa (+ColorJitter)"),
        ("aug_aggressive", "Agresywna (+Rotate+Perspective)"),
    ]
    opt_map = [
        ("opt_adam",      "Adam (lr=1e-4)"),
        ("opt_adamw_1e4", "AdamW (wd=1e-4)"),
        ("opt_adamw_1e3", "AdamW (wd=1e-3)"),
        ("opt_sgd",       "SGD (momentum=0.9)"),
    ]
    sched_map = [
        ("sched_cosine",   "CosineAnnealingWarmRestarts"),
        ("sched_plateau",  "ReduceLROnPlateau"),
        ("sched_onecycle", "OneCycleLR"),
        ("sched_step",     "StepLR (step=5, γ=0.1)"),
    ]
    dropout_map = [
        ("dropout_00", "p=0.0"),
        ("dropout_01", "p=0.1"),
        ("dropout_03", "p=0.3"),
        ("dropout_05", "p=0.5"),
    ]
    best_r_noisy = try_load("best_etap7_noisy")
    best_r_clean = try_load("best_etap7_clean")

    def load_group(mapping):
        rows = []
        for eid, name in mapping:
            entry = {"id": eid, "name": name, "noisy": None, "clean": None}
            for tag in ("noisy", "clean"):
                r = try_load(f"{eid}_{tag}")
                if r:
                    entry[tag] = r["metrics"]
            rows.append(entry)
        return rows

    d["etap7_experiments"] = {
        "augmentation": load_group(aug_map),
        "optimizer":    load_group(opt_map),
        "scheduler":    load_group(sched_map),
        "dropout":      load_group(dropout_map),
        "best_model": {
            "noisy": best_r_noisy["metrics"] if best_r_noisy else None,
            "clean": best_r_clean["metrics"] if best_r_clean else None,
        },
    }
    return d


def table_model_profile(d: dict):
    p = d.get("model_profile", {})
    rows = [
        ("Architektura bazowa",       "MobileNetV2 x0.75 (CIFAR-100)"),
        ("Liczba parametrów",         f"{p.get('total_params', 'N/A'):,}" if p.get('total_params') else "N/A"),
        ("Rozmiar pliku (FP32)",      f"{num(p.get('size_mb'))} MB"),
        ("Czas wnioskowania CPU",     f"{num(p.get('inference_cpu_ms'))} ms/obraz"),
        ("Czas wnioskowania GPU",     f"{num(p.get('inference_gpu_ms'))} ms/obraz"),
        ("Czas treningu jednej epoki",f"{num(p.get('epoch_time_s'))} s"),
        ("Szczytowe użycie VRAM",     f"{num(p.get('peak_memory_mb'), '.0f')} MB"),
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
    source = d.get("_source", "example")

    if source == "example":
        b = d["baseline"]["metrics"]
        variants = [("Z szumem", b)]
    else:
        variants = []
        for label, tag in [("Z szumem", "noisy"), ("Czyste", "clean")]:
            m = d["baseline"].get(tag)
            if m:
                variants.append((label, m))

    if not variants:
        return

    metric_labels = [
        ("Dokładność testowa (top-1)", "test_acc",        pct),
        ("F1-score (macro, 3 klasy)",  "test_f1",         pct),
        ("Rozmiar modelu",             "size_mb",         lambda v: f"{num(v)} MB"),
        ("Parametry (łącznie)",        "total_params",    lambda v: f"{int(v):,}" if v else "N/A"),
        ("Czas wnioskowania CPU",      "inference_cpu_ms", lambda v: f"{num(v)} ms/obraz"),
    ]

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Wyniki -- baseline}",
        r"\label{tab:baseline}",
    ]
    if len(variants) == 2:
        lines += [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"\textbf{Metryka} & \textbf{Z szumem} & \textbf{Czyste} \\",
            r"\midrule",
        ]
        for ml, key, fmt in metric_labels:
            v1 = fmt(variants[0][1].get(key))
            v2 = fmt(variants[1][1].get(key))
            lines.append(f"{ml} & {v1} & {v2} \\\\")
    else:
        lines += [
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"\textbf{Metryka} & \textbf{Wartość} \\",
            r"\midrule",
        ]
        for ml, key, fmt in metric_labels:
            lines.append(f"{ml} & {fmt(variants[0][1].get(key))} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("baseline", "\n".join(lines))


# ─── Table 3 : Pruning comparison ─────────────────────────────────────────────

def table_pruning(d: dict):
    source = d.get("_source", "example")

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Porównanie wyników przerzedzania – Etap~5.}",
        r"\label{tab:pruning}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Metoda} & \textbf{Rzadkość}",
        r"& \textbf{Trening} & \textbf{Acc.} & \textbf{F1} & \textbf{CPU [ms]} \\",
        r"\midrule",
    ]
    rows = []

    if source == "example":
        exps = d["pruning_experiments"]
        baseline = d["baseline"]["metrics"]
        rows.append(
            f"Odniesienie & 0\\% & Z szumem & {pct(baseline['test_acc'])} "
            f"& {pct(baseline['test_f1'])} & {num(baseline['inference_cpu_ms'])} \\\\"
        )
        prev_method = None
        for e in exps:
            m = e["metrics"]
            label = "Niestrukturalne L1" if e["method"] == "unstructured" else "Strukturalne LN"
            if prev_method is not None and e["method"] != prev_method:
                rows.append(r"\midrule")
            rows.append(
                f"{label} & {pct(e['sparsity'])} & Z szumem "
                f"& {pct(m['test_acc'])} & {pct(m['test_f1'])} & {num(m.get('inference_cpu_ms'))} \\\\"
            )
            prev_method = e["method"]
    else:
        bl = d["baseline"]
        for tag, label in [("noisy", "Z szumem"), ("clean", "Czyste")]:
            m = bl.get(tag)
            if m:
                rows.append(
                    f"Odniesienie & 0\\% & {label} & {pct(m.get('test_acc'))} "
                    f"& {pct(m.get('test_f1'))} & {num(m.get('inference_cpu_ms'))} \\\\"
                )
        rows.append(r"\midrule")

        prev_method = None
        for e in d["pruning_experiments"]:
            label = "Niestrukturalne L1" if e["method"] == "unstructured" else "Strukturalne LN"
            if prev_method is not None and e["method"] != prev_method:
                rows.append(r"\midrule")
            for tag, tl in [("noisy", "Z szumem"), ("clean", "Czyste")]:
                m = e.get(tag)
                if m:
                    rows.append(
                        f"{label} & {pct(e['sparsity'])} & {tl} "
                        f"& {pct(m.get('test_acc'))} & {pct(m.get('test_f1'))} "
                        f"& {num(m.get('inference_cpu_ms'))} \\\\"
                    )
            prev_method = e["method"]

    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("pruning", "\n".join(header + rows + footer))


# ─── Table 4 : Quantization ───────────────────────────────────────────────────

def table_quantization(d: dict):
    source = d.get("_source", "example")

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wyniki kwantyzacji modelu – Etap~6.}",
        r"\label{tab:quantization}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Metoda} & \textbf{Trening} & \textbf{Acc.} & \textbf{F1} "
        r"& \textbf{Rozmiar} \\",
        r"\midrule",
    ]
    rows = []

    if source == "example":
        baseline = d["baseline"]["metrics"]
        rows.append(
            f"Odniesienie (FP32) & Z szumem & {pct(baseline['test_acc'])} "
            f"& {pct(baseline['test_f1'])} & {num(baseline['size_mb'])}~MB \\\\"
        )
        for e in d["quantization_experiments"]:
            m = e["metrics"]
            rows.append(
                f"{esc(e['name'])} & Z szumem & {pct(m['test_acc'])} "
                f"& {pct(m['test_f1'])} & {num(m['size_mb'])}~MB \\\\"
            )
    else:
        bl = d["baseline"]
        for tag, label in [("noisy", "Z szumem"), ("clean", "Czyste")]:
            m = bl.get(tag)
            if m:
                rows.append(
                    f"Odniesienie (FP32) & {label} & {pct(m.get('test_acc'))} "
                    f"& {pct(m.get('test_f1'))} & {num(m.get('size_mb'))}~MB \\\\"
                )
        rows.append(r"\midrule")
        for e in d["quantization_experiments"]:
            for tag, label in [("noisy", "Z szumem"), ("clean", "Czyste")]:
                m = e.get(tag)
                if m:
                    rows.append(
                        f"{esc(e['name'])} & {label} & {pct(m.get('test_acc'))} "
                        f"& {pct(m.get('test_f1'))} & {num(m.get('size_mb'))}~MB \\\\"
                    )

    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("quantization", "\n".join(header + rows + footer))


# ─── Table 5 : Augmentation ───────────────────────────────────────────────────

def table_augmentation(d: dict):
    source = d.get("_source", "example")
    exps = d["etap7_experiments"]["augmentation"]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wpływ strategii augmentacji na dokładność – Etap~7.}",
        r"\label{tab:augmentation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Strategia}",
        r"& \textbf{Acc. (sz.)} & \textbf{F1 (sz.)}",
        r"& \textbf{Acc. (cz.)} & \textbf{F1 (cz.)} \\",
        r"\midrule",
    ]
    rows = []
    for e in exps:
        if source == "example":
            m = e["metrics"]
            rows.append(
                f"{e['name']} & {pct(m['test_acc'])} & {pct(m['test_f1'])} & N/A & N/A \\\\"
            )
        else:
            mn = e.get("noisy") or {}
            mc = e.get("clean") or {}
            rows.append(
                f"{e['name']} & {pct(mn.get('test_acc'))} & {pct(mn.get('test_f1'))} "
                f"& {pct(mc.get('test_acc'))} & {pct(mc.get('test_f1'))} \\\\"
            )
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("augmentation", "\n".join(header + rows + footer))


# ─── Table 6 : Hyperparameters grid ───────────────────────────────────────────

def table_hyperparams(d: dict):
    source = d.get("_source", "example")
    e7 = d["etap7_experiments"]

    groups = [
        ("Optymalizator",  e7["optimizer"]),
        ("Harmonogram LR", e7["scheduler"]),
        ("Dropout",        e7["dropout"]),
    ]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Przegląd hiperparametrów uczenia – Etap~7.}",
        r"\label{tab:hyperparams}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Grupa} & \textbf{Wariant}",
        r"& \textbf{Acc. (sz.)} & \textbf{F1 (sz.)}",
        r"& \textbf{Acc. (cz.)} & \textbf{F1 (cz.)} \\",
        r"\midrule",
    ]
    rows = []
    for group_name, exps in groups:
        for i, e in enumerate(exps):
            g_label = group_name if i == 0 else ""
            if source == "example":
                m = e["metrics"]
                rows.append(
                    f"{g_label} & {esc(e['name'])} & {pct(m['test_acc'])} & {pct(m['test_f1'])} "
                    f"& N/A & N/A \\\\"
                )
            else:
                mn = e.get("noisy") or {}
                mc = e.get("clean") or {}
                rows.append(
                    f"{g_label} & {esc(e['name'])} "
                    f"& {pct(mn.get('test_acc'))} & {pct(mn.get('test_f1'))} "
                    f"& {pct(mc.get('test_acc'))} & {pct(mc.get('test_f1'))} \\\\"
                )
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    # best model row
    rows.append(r"\midrule")
    best = e7["best_model"]
    best_desc = esc("AdamW + OneCycleLR + aug. std. + Dropout 0.1")
    if source == "example":
        bm = best["metrics"]
        rows.append(
            r"\textbf{Najlepszy} & \textbf{" + best_desc + r"} "
            r"& \textbf{" + pct(bm["test_acc"]) + r"} & \textbf{" + pct(bm["test_f1"]) + r"}"
            r" & N/A & N/A \\"
        )
    else:
        mn = best.get("noisy") or {}
        mc = best.get("clean") or {}
        rows.append(
            r"\textbf{Najlepszy} & \textbf{" + best_desc + r"} "
            r"& \textbf{" + pct(mn.get("test_acc")) + r"} & \textbf{" + pct(mn.get("test_f1")) + r"}"
            r" & \textbf{" + pct(mc.get("test_acc")) + r"} & \textbf{" + pct(mc.get("test_f1")) + r"} \\"
        )

    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("hyperparams", "\n".join(header + rows + footer))


# ─── Table 7 : Final comparison ───────────────────────────────────────────────

def table_final_comparison(d: dict):
    source = d.get("_source", "example")

    if source == "example":
        baseline  = d["baseline"]["metrics"]
        best_e7   = d["etap7_experiments"]["best_model"]["metrics"]
        best_opt  = d["etap7_experiments"]["best_optimized"]["metrics"]
        ptq       = d["quantization_experiments"][0]["metrics"]
        entries = [
            ("Odniesienie (Etap~4)",    "Z szumem", baseline["test_acc"], baseline["test_f1"],
             baseline["size_mb"], baseline["inference_cpu_ms"]),
            ("PTQ INT8 (Etap~6)",       "Z szumem", ptq["test_acc"], ptq["test_f1"],
             ptq["size_mb"], ptq["inference_cpu_ms"]),
            ("Najlepszy Etap~7",        "Z szumem", best_e7["test_acc"], best_e7["test_f1"],
             best_e7["size_mb"], best_e7["inference_cpu_ms"]),
            ("Najlepszy + Pruning+PTQ", "Z szumem", best_opt["test_acc"], best_opt["test_f1"],
             best_opt["size_mb"], best_opt["inference_cpu_ms"]),
        ]
    else:
        bl_n = d["baseline"].get("noisy") or {}
        bl_c = d["baseline"].get("clean") or {}
        ptq_n = next((e.get("noisy") for e in d["quantization_experiments"]
                      if e["id"] == "quant_ptq_dynamic"), None) or {}
        ptq_c = next((e.get("clean") for e in d["quantization_experiments"]
                      if e["id"] == "quant_ptq_dynamic"), None) or {}
        pq_n = next((e.get("noisy") for e in d["quantization_experiments"]
                     if e["id"] == "prune50_quant"), None) or {}
        pq_c = next((e.get("clean") for e in d["quantization_experiments"]
                     if e["id"] == "prune50_quant"), None) or {}
        best_n = d["etap7_experiments"]["best_model"].get("noisy") or {}
        best_c = d["etap7_experiments"]["best_model"].get("clean") or {}
        entries = [
            ("Odniesienie (Etap~4)",    "Z szumem", bl_n.get("test_acc"), bl_n.get("test_f1"),
             bl_n.get("size_mb"), bl_n.get("inference_cpu_ms")),
            ("Odniesienie (Etap~4)",    "Czyste",   bl_c.get("test_acc"), bl_c.get("test_f1"),
             bl_c.get("size_mb"), bl_c.get("inference_cpu_ms")),
            ("PTQ INT8 (Etap~6)",       "Z szumem", ptq_n.get("test_acc"), ptq_n.get("test_f1"),
             ptq_n.get("size_mb"), ptq_n.get("inference_cpu_ms")),
            ("PTQ INT8 (Etap~6)",       "Czyste",   ptq_c.get("test_acc"), ptq_c.get("test_f1"),
             ptq_c.get("size_mb"), ptq_c.get("inference_cpu_ms")),
            ("Najlepszy Etap~7",        "Z szumem", best_n.get("test_acc"), best_n.get("test_f1"),
             best_n.get("size_mb"), best_n.get("inference_cpu_ms")),
            ("Najlepszy Etap~7",        "Czyste",   best_c.get("test_acc"), best_c.get("test_f1"),
             best_c.get("size_mb"), best_c.get("inference_cpu_ms")),
            ("Pruning 50\\%+PTQ",       "Z szumem", pq_n.get("test_acc"), pq_n.get("test_f1"),
             pq_n.get("size_mb"), pq_n.get("inference_cpu_ms")),
            ("Pruning 50\\%+PTQ",       "Czyste",   pq_c.get("test_acc"), pq_c.get("test_f1"),
             pq_c.get("size_mb"), pq_c.get("inference_cpu_ms")),
        ]

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Zbiorcze porównanie wszystkich wariantów modelu.}",
        r"\label{tab:final-comparison}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Wariant} & \textbf{Trening} & \textbf{Acc.} & \textbf{F1} "
        r"& \textbf{Rozmiar} & \textbf{CPU [ms]} \\",
        r"\midrule",
    ]
    rows = []
    for name, training, acc, f1, size, cpu in entries:
        rows.append(
            f"{name} & {training} & {pct(acc)} & {pct(f1)} "
            f"& {num(size)}~MB & {num(cpu)} \\\\"
        )
    footer = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("final_comparison", "\n".join(header + rows + footer))


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating LaTeX tables...")
    d = load_data()
    src = d.get("_source", "?")
    print(f"  Data source: {src}")
    table_model_profile(d)
    table_baseline(d)
    table_pruning(d)
    table_quantization(d)
    table_augmentation(d)
    table_hyperparams(d)
    table_final_comparison(d)
    print(f"\nAll tables written to {TABLES_DIR}")
