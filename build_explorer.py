"""
Refresh the data embedded in the interactive explorer (docs/index.html).

  python build_explorer.py

The explorer is a single static page served by GitHub Pages. Its numbers and raw
outputs live in a <script type="application/json" id="data"> block, rebuilt here
from results/run1-3.json and the eval files, using the same format rules as
tally.py. Only the original three runs are included.
"""

from __future__ import annotations

import json
import os
import re

from tally import recount_contract, recount_format_any

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "docs", "index.html")
EVAL_FILES = {1: "data/eval.jsonl", 2: "data/eval_hard.jsonl", 3: "data/eval_hard.jsonl"}
DATA_BLOCK = re.compile(r'(<script type="application/json" id="data">)(.*?)(</script>)', re.S)


def load_jsonl(path: str) -> dict[str, dict]:
    with open(os.path.join(ROOT, path)) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def build_run(n: int) -> dict:
    with open(os.path.join(ROOT, "results", f"run{n}.json")) as f:
        res = json.load(f)
    evals = load_jsonl(EVAL_FILES[n])
    metrics = {}
    for variant, m in res["metrics"].items():
        records = res["raw"][variant]
        metrics[variant] = {**m, "format_any": recount_format_any(records), "contract": recount_contract(records)}
    prompts = []
    for i, rec in enumerate(res["raw"]["Base-low"]):
        ev = evals[rec["id"]]
        prompts.append({
            "id": rec["id"], "label": rec["label"], "cat": ev["category"], "prompt": ev["prompt"],
            "v": {v: {"d": res["raw"][v][i]["decisions"], "o": res["raw"][v][i]["outputs"]} for v in res["raw"]},
        })
    return {"n": n, "metrics": metrics, "prompts": prompts}


def main():
    data = {"runs": [build_run(n) for n in (1, 2, 3)]}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(PAGE) as f:
        page = f.read()
    page, count = DATA_BLOCK.subn(lambda m: m.group(1) + blob + m.group(3), page)
    if count != 1:
        raise SystemExit(f"expected one data block in {PAGE}, found {count}")
    with open(PAGE, "w") as f:
        f.write(page)
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
