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
    sparsity_levels = [10, 15, 20, 30, 50, 70, 80, 90, 95, 99]
    for p_type in ["unstruct", "struct"]:
        for p_sched in ["o", "s"]:
            for sp in sparsity_levels:
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
    for kind in ("infer", "ephemeral"):
        kind_exps = []
        for sp in [10, 30, 50, 70, 80, 90, 95, 99]:
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
        ("quant_ptq_dynamic", "PTQ dynamiczna (INT8)"),
        ("quant_qat",         "QAT (INT8)"),
        ("prune50_quant",     "Pruning 50\\% + PTQ INT8"),
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

    # --- ETAP 7 ---
    def load_group(mapping):
        rows = []
        for eid, name in mapping:
            entry = {"id": eid, "name": name, "noisy": None, "clean": None}
            for tag in ("noisy", "clean"):
                r = try_load(f"{eid}_{tag}")
                if r and "metrics" in r: entry[tag] = r["metrics"]
            rows.append(entry)
        return rows

    d["etap7_experiments"] = {
        "augmentation": load_group([("aug_none", "Brak"), ("aug_basic", "Podstawowa")]),
        "optimizer":    load_group([("opt_adam", "Adam"), ("opt_adamw_1e4", "AdamW")]),
        "scheduler":    load_group([("sched_cosine", "Cosine")]),
        "dropout":      load_group([("dropout_00", "p=0.0"), ("dropout_05", "p=0.5")]),
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
    _create_pruning_table(d, "unstruct", "pruning_unstructured",
                          "UNSTRUCTURED")

def table_pruning_structured(d: dict):
    _create_pruning_table(d, "struct", "pruning_structured",
                          "STRUCTURED")


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
                                  "Przerzedzanie efemeryczne (odwracalne) – Etap 5c.")


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
    print(f"\nAll tables written to {TABLES_DIR}")
