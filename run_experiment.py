"""
Clarify-or-Act: the full four-variant experiment on Inkling-Small.

Implements the dataset card's design:
  Base-low (0.2) | Base-high (0.9) | Fine-tuned-low (0.2) | Fine-tuned-high (0.9)
Each eval prompt is sampled 3 times per variant with the SAME system message,
temperature, and max_tokens across conditions.

It prints a metrics table and writes a results JSON with every raw output so the
semantic rubric can be audited by hand (or with an LLM grader); the card is clear
that the rubric should not be trusted to a lexical proxy.

Run presets (see results/README.md for what each run was):
  python run_experiment.py --run 1     # 40 examples, 6 epochs, easy 20-prompt eval
  python run_experiment.py --run 2     # 40 examples, 6 epochs, hard 16-prompt eval
  python run_experiment.py --run 3     # 52 examples, 3 epochs, hard 16-prompt eval (default)

Each run writes results/run<N>_<timestamp>.json (or --out <path>) with a "config"
block recording the exact knobs and data files used, then prints the tally.
Requires TINKER_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import random

import tinker
from tinker import types
from tinker_cookbook.supervised.common import compute_mean_nll

import clarify_common as cc

# ---- knobs shared by every run ----------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
RANK = 32
SAMPLES_PER_PROMPT = 3
MAX_TOKENS = 2048          # constant across all conditions (card requirement)
TEMPERATURE = 0.7          # >0 so the 3-sample consistency metric is meaningful
CONCURRENCY = 8            # generations run in parallel (raise/lower if rate-limited)

# ---- per-run presets: the one or two things that changed between runs --------
RUNS = {
    1: dict(epochs=6, train_file="data/train.jsonl",    eval_file="data/eval.jsonl",
            note="Baseline setup. Easy eval saturated at 100% for every variant."),
    2: dict(epochs=6, train_file="data/train.jsonl",    eval_file="data/eval_hard.jsonl",
            note="Same training as run 1; eval replaced with 16 borderline prompts."),
    3: dict(epochs=3, train_file="data/train_v3.jsonl", eval_file="data/eval_hard.jsonl",
            note="Same eval as run 2; +12 borderline-ACT training examples, epochs 6 -> 3."),
}
# ----------------------------------------------------------------------------


async def evaluate(sampling_client, renderer, eval_rows, effort: float) -> list[dict]:
    """Sample every prompt SAMPLES_PER_PROMPT times, concurrently (bounded)."""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(row_idx: int, sample_idx: int):
        async with sem:
            text, _ntok, _term = await cc.sample_once(
                sampling_client, renderer, cc.eval_messages(eval_rows[row_idx]),
                effort=effort, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
        return row_idx, text

    tasks = [one(r, s) for r in range(len(eval_rows)) for s in range(SAMPLES_PER_PROMPT)]
    outputs_by_row: dict[int, list[str]] = {r: [] for r in range(len(eval_rows))}
    for row_idx, text in await asyncio.gather(*tasks):
        outputs_by_row[row_idx].append(text)

    records = []
    for r, row in enumerate(eval_rows):
        required = row.get("required_question_elements") or row.get("required_answer_elements") or []
        disallowed = row.get("disallowed_behavior", [])
        outs = outputs_by_row[r]
        records.append({
            "id": row["id"], "label": row["label"],
            "decisions": [cc.parse_decision(t) for t in outs],
            "coverages": [cc.lexical_rubric_coverage(t, required, disallowed) for t in outs],
            "outputs": outs,
        })
    return records


def compute_metrics(records: list[dict]) -> dict:
    ask_hit = ask_tot = act_hit = act_tot = 0
    correct = total = 0
    fmt_fail = over_ask = under_ask = 0
    consistent = 0
    cov_sum = cov_n = 0.0
    over_ask_prompts = 0  # prompts (majority) where an ACT case became ASK
    contract_ok = 0       # outputs that follow the DECISION: + QUESTION:/ANSWER: contract

    for r in records:
        label = r["label"]
        decs = r["decisions"]
        for out in r["outputs"]:
            up = (out or "").upper()
            prefix = "QUESTION:" if label == "ASK" else "ANSWER:"
            if "DECISION:" in up and prefix in up:
                contract_ok += 1
        if len(set(decs)) == 1:
            consistent += 1
        # majority decision for per-prompt over-asking count
        maj = max(set(decs), key=decs.count)
        if label == "ACT" and maj == "ASK":
            over_ask_prompts += 1
        for d, c in zip(decs, r["coverages"]):
            total += 1
            cov_sum += c
            cov_n += 1
            if d is None:
                fmt_fail += 1
            if label == "ASK":
                ask_tot += 1
                if d == "ASK":
                    ask_hit += 1
                elif d == "ACT":
                    under_ask += 1
            else:
                act_tot += 1
                if d == "ACT":
                    act_hit += 1
                elif d == "ASK":
                    over_ask += 1
            if d == label:
                correct += 1

    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "decision_accuracy": pct(correct, total),
        "contract_compliance": pct(contract_ok, total),
        "clarification_recall": pct(ask_hit, ask_tot),
        "autonomy_rate": pct(act_hit, act_tot),
        "consistency": pct(consistent, len(records)),
        "rubric_coverage_approx": round(cov_sum / cov_n, 3) if cov_n else None,
        "format_failures": fmt_fail,
        "over_asking_samples": over_ask,
        "under_asking_samples": under_ask,
        "over_asking_prompts_majority": over_ask_prompts,
    }


def print_table(metrics_by_variant: dict):
    cols = ["Base-low", "Base-high", "FT-low", "FT-high"]
    rows = [
        ("Decision accuracy %", "decision_accuracy"),
        ("Format-contract %", "contract_compliance"),
        ("Clarification recall %", "clarification_recall"),
        ("Autonomy rate %", "autonomy_rate"),
        ("Consistency %", "consistency"),
        ("Rubric coverage (approx)", "rubric_coverage_approx"),
        ("Format failures", "format_failures"),
        ("Over-asking (samples)", "over_asking_samples"),
        ("Under-asking (samples)", "under_asking_samples"),
    ]
    w = 26
    print("\n" + "=" * (w + 4 * 12))
    print(f"{'metric':<{w}}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (w + 4 * 12))
    for label, key in rows:
        cells = "".join(f"{str(metrics_by_variant[c][key]):>12}" for c in cols)
        print(f"{label:<{w}}{cells}")
    print("=" * (w + 4 * 12))


def check_success_criteria(m: dict):
    print("\nSuccess criteria (from the dataset card):")

    def delta(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 1)

    rec_low = delta(m["FT-low"]["clarification_recall"], m["Base-low"]["clarification_recall"])
    rec_high = delta(m["FT-high"]["clarification_recall"], m["Base-high"]["clarification_recall"])
    aut_low = delta(m["FT-low"]["autonomy_rate"], m["Base-low"]["autonomy_rate"])
    aut_high = delta(m["FT-high"]["autonomy_rate"], m["Base-high"]["autonomy_rate"])

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    print(f"  1. Clarification recall +>=20pp vs same-effort base: "
          f"low {rec_low}pp [{verdict(rec_low is not None and rec_low >= 20)}], "
          f"high {rec_high}pp [{verdict(rec_high is not None and rec_high >= 20)}]")
    print(f"  2. Autonomy degrades <=5pp: "
          f"low {aut_low}pp [{verdict(aut_low is not None and aut_low >= -5)}], "
          f"high {aut_high}pp [{verdict(aut_high is not None and aut_high >= -5)}]")
    print(f"  3. <=1 held-out ACT prompt turned into clarification (FT-low): "
          f"{m['FT-low']['over_asking_prompts_majority']} "
          f"[{verdict(m['FT-low']['over_asking_prompts_majority'] <= 1)}]")
    ftlow, basehigh = m["FT-low"]["decision_accuracy"], m["Base-high"]["decision_accuracy"]
    print(f"  4. FT-low decision accuracy approaches/exceeds Base-high: "
          f"{ftlow}% vs {basehigh}% "
          f"[{verdict(ftlow is not None and basehigh is not None and ftlow >= basehigh - 2)}]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=int, choices=sorted(RUNS), default=3,
                   help="which run preset to execute (default 3)")
    p.add_argument("--epochs", type=int, help="override the preset's epoch count")
    p.add_argument("--train-file", help="override the preset's training file")
    p.add_argument("--eval-file", help="override the preset's eval file")
    p.add_argument("--out", help="output path (default results/run<N>_<timestamp>.json)")
    return p.parse_args()


async def main():
    args = parse_args()
    preset = dict(RUNS[args.run])
    epochs = args.epochs or preset["epochs"]
    train_file = args.train_file or preset["train_file"]
    eval_file = args.eval_file or preset["eval_file"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join("results", f"run{args.run}_{stamp}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Export it and re-run.")

    renderer, _tok = cc.build_renderer()
    train_rows = cc.load_jsonl(train_file)
    eval_rows = cc.load_jsonl(eval_file)

    config = {
        "run": args.run, "note": preset["note"], "started_utc": stamp,
        "model": cc.MODEL, "train_file": train_file, "n_train": len(train_rows),
        "eval_file": eval_file, "n_eval": len(eval_rows),
        "epochs": epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "rank": RANK,
        "samples_per_prompt": SAMPLES_PER_PROMPT, "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "efforts": cc.EFFORT, "system_message": cc.SYSTEM,
    }
    print(f"Run {args.run}: {preset['note']}")
    print(f"  train {train_file} ({len(train_rows)} ex), eval {eval_file} ({len(eval_rows)} prompts), "
          f"{epochs} epochs")

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(base_model=cc.MODEL, rank=RANK)

    # ---- BASELINE (before any training) --------------------------------
    print(f"[1/4] Baseline eval: {len(eval_rows)} prompts x {SAMPLES_PER_PROMPT} samples "
          f"at effort {cc.EFFORT['low']} and {cc.EFFORT['high']} ...")
    base_sampler = await service.create_sampling_client_async(base_model=cc.MODEL)
    base_low = await evaluate(base_sampler, renderer, eval_rows, cc.EFFORT["low"])
    base_high = await evaluate(base_sampler, renderer, eval_rows, cc.EFFORT["high"])

    # ---- TRAIN ---------------------------------------------------------
    print(f"[2/4] Fine-tuning LoRA on {len(train_rows)} examples "
          f"({epochs} epochs, batch {BATCH_SIZE}, lr {LEARNING_RATE}, rank {RANK}) ...")
    datums = cc.build_datums(renderer, train_rows)
    losses = []
    for epoch in range(epochs):
        random.shuffle(datums)
        # Tinker's forward_backward returns per-token logprobs, not a scalar loss.
        # Train loss = weighted mean NLL over assistant tokens across the whole epoch.
        epoch_logprobs, epoch_weights = [], []
        for start in range(0, len(datums), BATCH_SIZE):
            batch = datums[start:start + BATCH_SIZE]
            fb = await training_client.forward_backward_async(data=batch, loss_fn="cross_entropy")
            fb_res = await fb.result_async()
            epoch_logprobs += [out["logprobs"] for out in fb_res.loss_fn_outputs]
            epoch_weights += [d.loss_fn_inputs["weights"] for d in batch]
            opt = await training_client.optim_step_async(types.AdamParams(learning_rate=LEARNING_RATE))
            await opt.result_async()
        epoch_nll = compute_mean_nll(epoch_logprobs, epoch_weights)
        losses.append(round(float(epoch_nll), 4))
        print(f"      epoch {epoch + 1}/{epochs}  train mean NLL={epoch_nll:.4f}")

    # ---- FINE-TUNED ----------------------------------------------------
    print("[3/4] Fine-tuned eval at both effort levels ...")
    tuned_sampler = await training_client.save_weights_and_get_sampling_client_async()
    ft_low = await evaluate(tuned_sampler, renderer, eval_rows, cc.EFFORT["low"])
    ft_high = await evaluate(tuned_sampler, renderer, eval_rows, cc.EFFORT["high"])

    # ---- REPORT --------------------------------------------------------
    print("[4/4] Scoring ...")
    raw = {"Base-low": base_low, "Base-high": base_high, "FT-low": ft_low, "FT-high": ft_high}
    metrics = {name: compute_metrics(recs) for name, recs in raw.items()}

    print_table(metrics)
    check_success_criteria(metrics)

    config["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config["epoch_losses"] = losses
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "raw": raw}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path} (config, metrics, and every raw output).")
    print("Run `python tally.py` to fold it into the results table.")
    print("Note: 'rubric_coverage_approx' is a lexical proxy only; audit the "
          "semantic rubric on the saved outputs before trusting it.")


if __name__ == "__main__":
    asyncio.run(main())
