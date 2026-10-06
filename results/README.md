# Results: provenance

Three original runs, all executed on 20 August 2026 against `thinkingmachines/Inkling-Small`
through the Tinker API. The original result files contain only `metrics` and `raw` (every
model output). They predate the `config` block that `run_experiment.py` now writes, so the
settings below are recorded here. Any file written by the current runner carries its own
`config` and needs no entry.

| File | Run | Training data | Epochs | Eval | Changed vs. prior run | Original mtime (PT) |
|---|---|---|---|---|---|---|
| `run1.json` | 1 | `data/train.jsonl`, 40 examples (20 ASK / 20 ACT) | 6 | `data/eval.jsonl`, 20 prompts | baseline setup | 2026-08-20 03:47 |
| `run2.json` | 2 | `data/train.jsonl`, 40 examples | 6 | `data/eval_hard.jsonl`, 16 prompts | the eval only | 2026-08-20 04:52 |
| `run3.json` | 3 | `data/train_v3.jsonl`, 52 examples (+12 borderline-ACT) | 3 | `data/eval_hard.jsonl`, 16 prompts | the training only | 2026-08-20 05:14 |

Shared across all runs: LoRA rank 32, learning rate 1e-4, batch size 8, three samples per
prompt, temperature 0.7, 2048-token budget, thinking effort 0.2 (low) and 0.9 (high), same
system message in every condition. `data/train_original.jsonl` is a byte-identical copy of
`data/train.jsonl`, kept because `build_train_v3.py` reads it.

## Metric notes

- **Two format metrics.** `contract_compliance` (the runner's metric, used in the write-up)
  counts an output only if it has `DECISION:` plus the prefix the label calls for:
  `QUESTION:` for ASK prompts, `ANSWER:` for ACT prompts. It therefore falls when the model
  over-asks. `tally.py` also reports pure format adherence, any `ANSWER:`/`QUESTION:` line
  regardless of decision, which the tune pushed to 96–100% in every run.
- **Run 1 has no `contract_compliance` field.** The metric was added for Run 2. `tally.py`
  recounts both format metrics from the raw outputs for every run with the same rule, so
  the columns are comparable. For Run 1 the base model produced the tag in 0 of 120
  outputs; the fine-tuned model in 118 of 120 (label-matched: 119 of 120).
- **`rubric_coverage_approx`** is a keyword-overlap proxy for answer quality, not a grade.
  No conclusion in the write-up depends on it.
- **A `None` decision** means the output had no parseable `DECISION:` first line. These are
  counted as `format_failures` and score as wrong for decision accuracy.

## Reproducing

```bash
export TINKER_API_KEY="..."
python run_experiment.py --run 1    # or 2, or 3
python tally.py
```

Each re-run writes `results/run<N>_<timestamp>.json` next to the originals and the tally
picks it up automatically. Sampling is at temperature 0.7 and LoRA training is not
seeded, so expect metrics to move by a few points between re-runs. The noise floor across
the original runs on the identical hard eval was about 4 points.
