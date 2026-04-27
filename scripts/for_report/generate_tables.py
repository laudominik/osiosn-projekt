from pathlib import Path
import json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR  = Path(__file__).resolve().parents[2] / "report" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def esc(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")

def pct(v) -> str:
    if v is None: return "N/A"
    return f"{v*100:.1f}\\%"

def pct_coarse(v) -> str:
    if v is None: return "N/A"
    return f"{v*100:.1f}\\%"

def num(v, fmt=".2f") -> str:
    if v is None: return "N/A"
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

def load_data() -> dict:
    d: dict = {"_source": "actual"}

    # --- BASELINE ---
    baseline_noisy = try_load("baseline_noisy")
    baseline_clean = try_load("baseline_clean")
    d["baseline"] = {
        "noisy": baseline_noisy["metrics"] if baseline_noisy else None,
        "clean": baseline_clean["metrics"] if baseline_clean else None,
    }

    for tag in ("noisy", "clean"):
        b = try_load(f"baseline_{tag}")
        if b and "metrics" in b:
            d["model_profile"] = {k: b["metrics"].get(k) for k in
                  ("total_params", "size_mb", "size_zip_mb", "inference_cpu_ms",
                   "inference_gpu_ms", "epoch_time_s", "peak_memory_mb")}
            break

    # --- PRUNING (trained) ---
    prune_exps = []
    # Unstructured: runs at high sparsity levels
    unstruct_levels = [50, 70, 80, 90, 95, 99]
    # Structured: runs at lower sparsity levels
    struct_levels   = [10, 15, 20, 30, 50]
    sparsity_map = {"unstruct": unstruct_levels, "struct": struct_levels}

    for p_type in ["unstruct", "struct"]:
        for p_sched in ["o", "s"]:
            for sp in sparsity_map[p_type]:
                entry = {"type": p_type, "schedule": p_sched, "noisy": None, "clean": None}
                found = False
                for tag in ("noisy", "clean"):
                    r = try_load(f"prune_{p_type}_{p_sched}_{sp:02d}_{tag}")
                    if r and "metrics" in r:
                        entry[tag] = r["metrics"]
                        entry["sparsity"] = r["metrics"].get("sparsity_target", sp / 100)
                        found = True
                if found:
                    prune_exps.append(entry)
    d["pruning_experiments"] = prune_exps

    # --- INFERENCE-TIME & EPHEMERAL PRUNING ---
    # inference_time.py uses: [0.10, 0.15, 0.25, 0.30, 0.50, 0.7, 0.9]
    infer_levels    = [10, 15, 25, 30, 50, 70, 90]
    ephemeral_levels = [10, 15, 25, 30, 50, 70, 90]
    levels_map = {"infer": infer_levels, "ephemeral": ephemeral_levels}

    for kind in ("infer", "ephemeral"):
        kind_exps = []
        for sp in levels_map[kind]:
            entry = {"sparsity": sp / 100, "noisy": None, "clean": None}
            found = False
            for tag in ("noisy", "clean"):
                r = try_load(f"prune_{kind}_{sp:02d}_{tag}")
                if r and "metrics" in r:
                    entry[tag] = r["metrics"]
                    entry["sparsity"] = r["metrics"].get("sparsity_target", sp / 100)
                    found = True
            if found:
                kind_exps.append(entry)
        d[f"pruning_{kind}"] = kind_exps

    # --- QUANTIZATION ---
    quant_exps = []
    quant_ids = [
        ("quant_ptq_dynamic", "PTQ do INT8"),
        ("quant_qat_fx",      "QAT do INT8"),
    ]
    for qid, qname in quant_ids:
        entry = {"id": qid, "name": qname, "noisy": None, "clean": None}
        for tag in ("noisy", "clean"):
            r = try_load(f"{qid}_{tag}")
            if r and "metrics" in r:
                entry[tag] = r["metrics"]
        if entry["noisy"] or entry["clean"]:
            quant_exps.append(entry)
    d["quantization_experiments"] = quant_exps

    # --- MODEL FINALNY ---
    final_n = try_load("final_noisy")
    final_c = try_load("final_clean")
    d["final"] = {
        "noisy": final_n["metrics"] if final_n else None,
        "clean": final_c["metrics"] if final_c else None,
    }

    # --- ETAP 7: hyperparameter optimisation ---
    def load_group(mapping):
        rows = []
        for eid, name in mapping:
            entry = {"id": eid, "name": name, "noisy": None, "clean": None}
            for tag in ("noisy", "clean"):
                r = try_load(f"{eid}_{tag}")
                if r and "metrics" in r:
                    entry[tag] = r["metrics"]
            rows.append(entry)
        return rows

    d["etap7_experiments"] = {
        "augmentation": load_group([
            ("aug_none",       "Brak"),
            ("aug_basic",      "Podstawowa"),
            ("aug_standard",   "Standardowa"),
            ("aug_aggressive", "Agresywna"),
        ]),
        "optimizer": load_group([
            ("opt_adam",      "Adam"),
            ("opt_adamw_1e4", "AdamW 1e-4"),
            ("opt_adamw_1e3", "AdamW 1e-3"),
            ("opt_sgd",       "SGD"),
        ]),
        "scheduler": load_group([
            ("sched_cosine",   "Cosine"),
            ("sched_plateau",  "Plateau"),
            ("sched_onecycle", "OneCycle"),
            ("sched_step",     "Step"),
        ]),
        "dropout": load_group([
            ("dropout_00", "p=0.0"),
            ("dropout_01", "p=0.1"),
            ("dropout_03", "p=0.3"),
            ("dropout_05", "p=0.5"),
        ]),
        "best_model": {
            "noisy": (try_load("best_etap7_noisy") or {}).get("metrics"),
            "clean": (try_load("best_etap7_clean") or {}).get("metrics"),
        },
    }
    return d


def table_model_profile(d: dict):
    p = d.get("model_profile", {})
    rows = [
        ("Liczba parametrów",      f"{p.get('total_params', 'N/A'):,}" if p.get("total_params") else "N/A"),
        ("Rozmiar pliku (Pth)",    f"{num(p.get('size_mb'))} MB"),
        ("Rozmiar pliku (Zipped)", f"{num(p.get('size_zip_mb'))} MB"),
        ("Czas wnioskowania CPU",  f"{num(p.get('inference_cpu_ms'))} ms/obraz"),
        ("Szczytowe użycie VRAM",  f"{num(p.get('peak_memory_mb'), '.0f')} MB"),
    ]
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Charakterystyka modelu bazowego.}", r"\label{tab:model-profile}",
        r"\begin{tabular}{ll}", r"\toprule",
        r"\textbf{Metryka} & \textbf{Wartość} \\", r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("model_profile", "\n".join(lines))


def table_baseline(d: dict):
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{Wyniki modelu odniesienia (Baseline).}",
        r"\label{tab:baseline}", r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Metryka} & \textbf{Z szumem} & \textbf{Czyste} \\", r"\midrule",
    ]
    bl_n = d.get("baseline", {}).get("noisy") or {}
    bl_c = d.get("baseline", {}).get("clean") or {}
    for label, key, fmt in [
        ("Acc. testowe",        "test_acc",          pct),
        ("F1-score",            "test_f1",           pct),
        ("Rozmiar [MB]",        "size_mb",           lambda x: num(x)),
        ("Rozmiar ZIP [MB]",    "size_zip_mb",       lambda x: num(x)),
        ("Inferencja CPU [ms]", "inference_cpu_ms",  lambda x: num(x)),
    ]:
        lines.append(f"{label} & {fmt(bl_n.get(key))} & {fmt(bl_c.get(key))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("baseline", "\n".join(lines))


def _create_pruning_table(d: dict, p_type: str, table_name: str, caption: str):
    exps = [e for e in d.get("pruning_experiments", []) if e.get("type") == p_type]
    if not exps:
        return
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + table_name + r"}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"\textbf{Harm.} & \textbf{Sparsity} & \textbf{Trening} & \textbf{Acc.} & \textbf{F1} & \textbf{CPU[ms]} & \textbf{ZIP[MB]} & \textbf{PTH[MB]} \\",
        r"\midrule",
    ]
    sched_map = {"o": "ONESH", "s": "SCHED"}
    exps = sorted(exps, key=lambda x: (x["schedule"], x.get("sparsity", 0)))
    prev_sched = None
    for e in exps:
        sched_str = sched_map.get(e["schedule"], e["schedule"])
        if prev_sched is not None and e["schedule"] != prev_sched:
            lines.append(r"\midrule")
        for tag, label in [("noisy", "Szum"), ("clean", "Czyste")]:
            m = e.get(tag)
            if m:
                lines.append(
                    f"{sched_str} & {pct(e.get('sparsity'))} & {label} "
                    f"& {pct(m.get('test_acc'))} & {pct(m.get('test_f1'))} "
                    f"& {num(m.get('inference_cpu_ms'))} & {num(m.get('size_zip_mb'))} & {num(m.get('size_mb'))} \\\\"
                )
        prev_sched = e["schedule"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table(table_name, "\n".join(lines))


def table_pruning_unstructured(d: dict):
    _create_pruning_table(d, "unstruct", "pruning_unstructured", "UNSTRUCTURED")

def table_pruning_structured(d: dict):
    _create_pruning_table(d, "struct", "pruning_structured", "STRUCTURED")


def _create_inference_prune_table(d: dict, key: str, table_name: str, caption: str):
    exps = d.get(key, [])
    if not exps:
        return
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + table_name + r"}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{Sparsity} & \textbf{Trening} & \textbf{Acc.} & \textbf{F1} & \textbf{CPU[ms]} & \textbf{ZIP[MB]} \\",
        r"\midrule",
    ]
    for e in sorted(exps, key=lambda x: x.get("sparsity", 0)):
        for tag, label in [("noisy", "Szum"), ("clean", "Czyste")]:
            m = e.get(tag)
            if m:
                lines.append(
                    f"{pct(e.get('sparsity'))} & {label} "
                    f"& {pct(m.get('test_acc'))} & {pct(m.get('test_f1'))} "
                    f"& {num(m.get('inference_cpu_ms'))} & {num(m.get('size_zip_mb'))} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table(table_name, "\n".join(lines))


def table_pruning_inference_time(d: dict):
    _create_inference_prune_table(d, "pruning_infer", "pruning_inference_time",
                                  "INFERENCE")

def table_pruning_ephemeral(d: dict):
    _create_inference_prune_table(d, "pruning_ephemeral", "pruning_ephemeral",
                                  "Przerzedzanie efemeryczne (odwracalne).")


def table_quantization(d: dict):
    exps = d.get("quantization_experiments", [])
    if not exps:
        return
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{Wyniki kwantyzacji – Etap 6.}",
        r"\label{tab:quantization}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"\textbf{Metoda} & \textbf{Trening} & \textbf{Acc.} & \textbf{F1} & \textbf{CPU[ms]} & \textbf{ZIP[MB]} & \textbf{Size[MB]} \\",
        r"\midrule",
    ]
    for e in exps:
        for tag, label in [("noisy", "Szum"), ("clean", "Czyste")]:
            m = e.get(tag)
            if m:
                lines.append(
                    f"{esc(e['name'])} & {label} "
                    f"& {pct(m.get('test_acc'))} & {pct(m.get('test_f1'))} "
                    f"& {num(m.get('inference_cpu_ms'))} & {num(m.get('size_zip_mb'))} "
                    f"& {num(m.get('size_mb'))} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("quantization", "\n".join(lines))


def _hyperopt_table(rows: list[dict], table_name: str, caption: str, label: str):
    """Generic two-column (noisy/clean) hyperopt comparison table."""
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"\textbf{Wariant} & \multicolumn{2}{c}{\textbf{Z szumem}} & \multicolumn{2}{c}{\textbf{Czyste}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & \textbf{Acc.} & \textbf{F1} & \textbf{Acc.} & \textbf{F1} \\",
        r"\midrule",
    ]
    for e in rows:
        mn = e.get("noisy") or {}
        mc = e.get("clean") or {}
        lines.append(
            f"{esc(e['name'])} "
            f"& {pct_coarse(mn.get('test_acc'))} & {pct_coarse(mn.get('test_f1'))} "
            f"& {pct_coarse(mc.get('test_acc'))} & {pct_coarse(mc.get('test_f1'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table(table_name, "\n".join(lines))


def table_hyperopt_augmentation(d: dict):
    rows = d.get("etap7_experiments", {}).get("augmentation", [])
    _hyperopt_table(rows, "hyperopt_augmentation",
                    "Porównanie strategii augmentacji danych.",
                    "hyperopt-aug")

def table_hyperopt_optimizer(d: dict):
    rows = d.get("etap7_experiments", {}).get("optimizer", [])
    _hyperopt_table(rows, "hyperopt_optimizer",
                    "Porównanie optymalizatorów.",
                    "hyperopt-opt")

def table_hyperopt_scheduler(d: dict):
    rows = d.get("etap7_experiments", {}).get("scheduler", [])
    _hyperopt_table(rows, "hyperopt_scheduler",
                    "Porównanie harmonogramów uczenia.",
                    "hyperopt-sched")

def table_hyperopt_dropout(d: dict):
    rows = d.get("etap7_experiments", {}).get("dropout", [])
    _hyperopt_table(rows, "hyperopt_dropout",
                    "Wpływ współczynnika dropout.",
                    "hyperopt-dropout")

def table_hyperopt_best(d: dict):
    best = d.get("etap7_experiments", {}).get("best_model", {})
    mn = best.get("noisy") or {}
    mc = best.get("clean") or {}
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Najlepszy model po optymalizacji hiperparametrów (Etap 7).}",
        r"\label{tab:hyperopt-best}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Metryka} & \textbf{Z szumem} & \textbf{Czyste} \\", r"\midrule",
    ]
    for label, key, fmt in [
        ("Acc. testowe",        "test_acc",         pct),
        ("F1-score",            "test_f1",          pct),
    ]:
        lines.append(f"{label} & {fmt(mn.get(key))} & {fmt(mc.get(key))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("hyperopt_best", "\n".join(lines))


def table_final_comparison(d: dict):
    """Zestawienie: baseline vs PTQ vs QAT vs model finalny."""
    bl_n = d.get("baseline", {}).get("noisy") or {}
    bl_c = d.get("baseline", {}).get("clean") or {}
    qe   = {e["id"]: e for e in d.get("quantization_experiments", [])}
    ptq_n = (qe.get("quant_ptq_dynamic") or {}).get("noisy") or {}
    ptq_c = (qe.get("quant_ptq_dynamic") or {}).get("clean") or {}
    qat_n = (qe.get("quant_qat_fx") or {}).get("noisy") or {}
    qat_c = (qe.get("quant_qat_fx") or {}).get("clean") or {}
    fn_n  = d.get("final", {}).get("noisy") or {}
    fn_c  = d.get("final", {}).get("clean") or {}

    rows = [
        ("Baseline",               bl_n,  bl_c),
        ("\\textbf{Finalny}", fn_n,  fn_c),
    ]

    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{Zestawienie końcowe: baseline vs modele skwantyzowane vs model finalny.}",
        r"\label{tab:final-comparison}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"\textbf{Model} & \multicolumn{2}{c}{\textbf{Acc.}} "
        r"& \multicolumn{2}{c}{\textbf{F1}} "
        r"& \textbf{CPU [ms]} & \textbf{PTH [MB]} & \textbf{ZIP [MB]} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & \textbf{Szum} & \textbf{Czyste} & \textbf{Szum} & \textbf{Czyste} & & \\",
        r"\midrule",
    ]
    for name, mn, mc in rows:
        lines.append(
            f"{name} & {pct(mn.get('test_acc'))} & {pct(mc.get('test_acc'))} "
            f"& {pct(mn.get('test_f1'))} & {pct(mc.get('test_f1'))} "
            f"& {num(mn.get('inference_cpu_ms'))} & {num(mn.get('size_mb'))} & {num(mn.get('size_zip_mb'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("final_comparison", "\n".join(lines))


if __name__ == "__main__":
    print("Generating LaTeX tables...")
    d = load_data()
    table_model_profile(d)
    table_baseline(d)
    table_pruning_unstructured(d)
    table_pruning_structured(d)
    table_pruning_inference_time(d)
    table_pruning_ephemeral(d)
    table_quantization(d)
    table_hyperopt_augmentation(d)
    table_hyperopt_optimizer(d)
    table_hyperopt_scheduler(d)
    table_hyperopt_dropout(d)
    table_hyperopt_best(d)
    table_final_comparison(d)
    print(f"\nAll tables written to {TABLES_DIR}")
