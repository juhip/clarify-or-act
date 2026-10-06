"""
Tally every results file in results/ into the tables used in the write-up.

  python tally.py                 # print tables and write results/TALLY.md
  python tally.py --no-write      # print only

Reads results/run*.json. The three original files (run1/run2/run3.json) have no
"config" block; their settings come from results/README.md and are filled in
from the RUNS presets by run number. Re-runs written by run_experiment.py carry
their own config.

Two format metrics are reported, because they answer different questions:

  * contract (label-matched): the output carries DECISION: plus the prefix the
    LABEL calls for (QUESTION: for ASK prompts, ANSWER: for ACT prompts). This is
    the runner's `contract_compliance` metric and the number used in the
    write-up. It drops when the model over-asks, because an unwanted QUESTION:
    does not count.
  * format (any tag): the output carries an ANSWER: or QUESTION: line at all,
    regardless of the decision. This is pure format adherence.

Run 1 predates `contract_compliance`, so both numbers are recounted from the raw
outputs for every file, with the same rule, so the columns are comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PRESETS = {
    1: dict(epochs=6, train_file="data/train.jsonl", eval_file="data/eval.jsonl", n_train=40, n_eval=20),
    2: dict(epochs=6, train_file="data/train.jsonl", eval_file="data/eval_hard.jsonl", n_train=40, n_eval=16),
    3: dict(epochs=3, train_file="data/train_v3.jsonl", eval_file="data/eval_hard.jsonl", n_train=52, n_eval=16),
}

VARIANTS = ["Base-low", "Base-high", "FT-low", "FT-high"]
TAG = re.compile(r"^(ANSWER|QUESTION):", re.M)


def recount_format_any(records: list[dict]) -> float:
    """Share of outputs that carry an ANSWER: or QUESTION: line, whatever the decision."""
    outs = [o for r in records for o in r["outputs"]]
    return round(100 * sum(1 for o in outs if TAG.search(o or "")) / len(outs), 1) if outs else None


def recount_contract(records: list[dict]) -> float:
    """Share of outputs with DECISION: plus the prefix the label calls for (the runner's rule)."""
    ok = n = 0
    for r in records:
        prefix = "QUESTION:" if r["label"] == "ASK" else "ANSWER:"
        for o in r["outputs"]:
            up = (o or "").upper()
            n += 1
            if "DECISION:" in up and prefix in up:
                ok += 1
    return round(100 * ok / n, 1) if n else None


def load_all() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "run*.json"))):
        name = os.path.basename(path)
        m = re.match(r"run(\d)", name)
        if not m:
            continue
        data = json.load(open(path, encoding="utf-8"))
        run_no = int(m.group(1))
        cfg = data.get("config") or {**PRESETS[run_no], "run": run_no, "original": True}
        metrics = data["metrics"]
        for v in VARIANTS:
            metrics[v]["contract_recount"] = recount_contract(data["raw"][v])
            metrics[v]["format_any"] = recount_format_any(data["raw"][v])
        runs.append({"file": name, "run": run_no, "config": cfg, "metrics": metrics,
                     "is_original": "config" not in data})
    return runs


def fmt(a, b):
    """'base → ft' cell."""
    return f"{a:g} → {b:g}%" if a is not None and b is not None else "n/a"


def pair(m, key, effort):
    base = m[f"Base-{effort}"][key]
    ft = m[f"FT-{effort}"][key]
    return base, ft


def table_per_run(run) -> str:
    m = run["metrics"]
    c = run["config"]
    rows = [
        ("Decision accuracy", "decision_accuracy"),
        ("Clarification recall (asked when it should)", "clarification_recall"),
        ("Autonomy (acted when it should)", "autonomy_rate"),
        ("Format contract, label-matched (as in write-up)", "contract_recount"),
        ("Format, any ANSWER:/QUESTION: line", "format_any"),
        ("Consistency across 3 samples", "consistency"),
    ]
    out = [f"### Run {run['run']} ({run['file']})", "",
           f"{c.get('n_train')} training examples, {c.get('epochs')} epochs, eval `{c.get('eval_file')}` "
           f"({c.get('n_eval')} prompts), {'original Aug 2026 run' if run['is_original'] else 're-run ' + str(c.get('started_utc', ''))}.",
           "", "| Metric | Low effort (base → FT) | High effort (base → FT) |", "|---|---|---|"]
    for label, key in rows:
        out.append(f"| {label} | {fmt(*pair(m, key, 'low'))} | {fmt(*pair(m, key, 'high'))} |")
    fails = ", ".join(f"{v} {m[v]['format_failures']}" for v in VARIANTS)
    out.append(f"\nFormat failures (no parseable DECISION line): {fails}.")
    return "\n".join(out)


def table_side_by_side(runs) -> str:
    """The write-up's summary table: one column per run, cells 'base → FT' averaged over effort."""
    def avg_cell(m, key):
        b = [m[f"Base-{e}"][key] for e in ("low", "high") if m[f"Base-{e}"][key] is not None]
        f = [m[f"FT-{e}"][key] for e in ("low", "high") if m[f"FT-{e}"][key] is not None]
        if not b or not f:
            return "n/a"
        bl, fl = min(b), min(f)
        bh, fh = max(b), max(f)
        bs = f"{bl:g}" if bl == bh else f"{bl:g}–{bh:g}"
        fs = f"{fl:g}" if fl == fh else f"{fl:g}–{fh:g}"
        return f"{bs} → {fs}%"

    rows = [
        ("Decision accuracy", "decision_accuracy"),
        ("Clarification recall", "clarification_recall"),
        ("Autonomy", "autonomy_rate"),
        ("Format contract (label-matched, as in write-up)", "contract_recount"),
        ("Format, any ANSWER:/QUESTION: line", "format_any"),
    ]
    hdr = "| Metric | " + " | ".join(f"Run {r['run']}" + ("" if r["is_original"] else " (re-run)") for r in runs) + " |"
    sep = "|---|" + "---|" * len(runs)
    out = [hdr, sep]
    for label, key in rows:
        out.append(f"| {label} | " + " | ".join(avg_cell(r["metrics"], key) for r in runs) + " |")
    out.append("\nCells show base → fine-tuned. Where low- and high-effort variants differ, the range is given.")
    out.append("The label-matched contract metric counts an output only if its prefix matches the expected "
               "decision, so it falls when the model over-asks. The second format row counts any "
               "ANSWER:/QUESTION: line and isolates pure format adherence.")
    return "\n".join(out)


def lift_table(runs) -> str:
    """Fine-tuned minus base, per effort level, for the runs on the hard eval."""
    hard = [r for r in runs if "hard" in str(r["config"].get("eval_file", ""))]
    if not hard:
        return ""
    out = ["| Lift (FT − base, points) | " + " | ".join(f"Run {r['run']} low / high" for r in hard) + " |",
           "|---|" + "---|" * len(hard)]
    for label, key in [("Autonomy", "autonomy_rate"), ("Decision accuracy", "decision_accuracy"),
                       ("Clarification recall", "clarification_recall"),
                       ("Format contract, label-matched", "contract_recount")]:
        cells = []
        for r in hard:
            m = r["metrics"]
            d = [round(m[f"FT-{e}"][key] - m[f"Base-{e}"][key], 1) for e in ("low", "high")]
            cells.append(f"{d[0]:+g} / {d[1]:+g}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def noise_note(runs) -> str:
    hard = [r for r in runs if "hard" in str(r["config"].get("eval_file", ""))]
    if len(hard) < 2:
        return ""
    vals = sorted({r["metrics"][v]["autonomy_rate"] for r in hard for v in ("Base-low", "Base-high")})
    n = hard[0]["config"].get("n_eval", 16) * 3
    return (f"Noise floor: the untrained model's autonomy on the identical hard eval ranged "
            f"{vals[0]:g}–{vals[-1]:g}% across runs. Each cell is {n} samples, so one sample moves a "
            f"metric about {100 / n:.1f} points; treat differences under ~{2 * round(100 / n):g} points as noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    runs = load_all()
    if not runs:
        raise SystemExit("No results/run*.json files found.")

    parts = ["# Results tally", "", "Generated by `python tally.py` from every `results/run*.json`.", "",
             "## Side by side", "", table_side_by_side(runs), ""]
    lt = lift_table(runs)
    if lt:
        parts += ["## Lift on the hard eval", "", lt, ""]
    nn = noise_note(runs)
    if nn:
        parts += [nn, ""]
    parts += ["## Per run", ""]
    for r in runs:
        parts += [table_per_run(r), ""]
    text = "\n".join(parts)
    print(text)
    if not args.no_write:
        path = os.path.join(RESULTS_DIR, "TALLY.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
