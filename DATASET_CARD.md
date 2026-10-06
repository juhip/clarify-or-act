# Clarify-or-Act: A Micro-Dataset for Inkling-Small

## Objective

Test whether a small supervised LoRA fine-tune can improve Inkling-Small's ability to ask for missing critical information while preserving its ability to act directly on well-specified tasks.

The desired behavior is not general cautiousness. It is calibrated initiative:

- `ASK` when missing information could materially change the answer or authorize an external action.
- `ACT` when the user has supplied enough information to complete the task safely and correctly.

## Contents

Original data (Run 1):

- `data/train.jsonl`: 40 balanced supervised examples: 20 `ASK`, 20 `ACT`.
- `data/eval.jsonl`: 20 held-out prompts: 10 `ASK`, 10 `ACT`, with rubric elements.

Added for Runs 2 and 3 (see the sections at the end of this card):

- `data/eval_hard.jsonl`: 16 borderline held-out prompts: 8 `ASK`, 8 `ACT`.
- `data/train_v3.jsonl`: the original 40 plus 12 borderline-`ACT` examples, 52 total: 20 `ASK`, 32 `ACT`.
- `data/train_original.jsonl`: byte-identical copy of `data/train.jsonl`, read by `build_train_v3.py`.

Both splits cover ten categories: analytics, scheduling, travel, software, model selection, research, communication, product design, finance, and hiring/operations.

## Training schema

Each training line contains:

```json
{
  "id": "train_001",
  "split": "train",
  "category": "analytics",
  "label": "ASK",
  "ambiguity_type": "missing_metric_definition_and_data",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "DECISION: ASK\nQUESTION: ..."}
  ]
}
```

The `messages` array is the material used for supervised fine-tuning. The other fields are metadata for analysis.

## Evaluation schema

Each evaluation line contains the prompt, expected decision, and semantic rubric:

```json
{
  "id": "eval_001",
  "label": "ASK",
  "prompt": "...",
  "required_question_elements": ["..."],
  "disallowed_behavior": ["..."]
}
```

For `ACT` examples, the rubric field is `required_answer_elements`.

## Recommended output contract

Ask the model to use exactly one of these forms:

```text
DECISION: ASK
QUESTION: <one concise clarification containing only material missing information>
```

```text
DECISION: ACT
ANSWER: <the completed task>
```

Suggested system message:

```text
Decide whether the request contains enough information to complete correctly and safely. Ask only when missing information could materially change the result or when an external action lacks authorization or a resolved target. Otherwise act directly. Begin with exactly `DECISION: ASK` or `DECISION: ACT`. If asking, ask one compact compound question. Do not invent missing facts.
```

Use the same system message for baseline and fine-tuned evaluation. Do not include it only in one condition.

## Experimental design

Use `thinkingmachines/Inkling-Small` and compare:

| Variant | Training | Thinking effort |
| --- | --- | ---: |
| Base-low | None | 0.2 |
| Base-high | None | 0.9 |
| Fine-tuned-low | SFT LoRA | 0.2 |
| Fine-tuned-high | SFT LoRA | 0.9 |

Run three samples per evaluation prompt. Keep temperature, maximum tokens, system message, and prompt rendering constant across conditions.

## Primary metrics

1. **Decision accuracy:** exact match between the first-line decision and `label`.
2. **Clarification recall:** correct `ASK` decisions divided by all expected `ASK` examples.
3. **Autonomy rate:** correct `ACT` decisions divided by all expected `ACT` examples.
4. **Consistency:** proportion of prompts receiving the same decision across all three samples.
5. **Semantic rubric coverage:** percentage of required elements present, minus failures triggered by `disallowed_behavior`.

Decision accuracy must be reported separately from answer quality. A model can make the right decision but ask a poor question or produce an incorrect answer.

## Proposed success criteria

- Clarification recall improves by at least 20 percentage points over the same-effort baseline.
- Autonomy rate degrades by no more than 5 percentage points.
- No more than one held-out `ACT` prompt is converted into unnecessary clarification.
- Fine-tuned-low approaches or exceeds Base-high decision accuracy.

These thresholds are hypotheses for a micro-experiment, not statistically validated production gates.

## Failure taxonomy

Review every error and assign one primary cause:

- `over_asking`: asks despite sufficient information.
- `under_asking`: acts while a material variable is missing.
- `question_bloat`: asks for nonessential information.
- `invented_fact`: silently fills a missing fact.
- `wrong_action`: chooses `ACT` but completes the task incorrectly.
- `format_failure`: does not emit the required decision prefix.

## Data limitations

- The dataset is synthetic and small.
- The label policy reflects one definition of calibrated initiative; reasonable users may disagree on borderline cases.
- The categories are broad but not representative of production traffic.
- Several rubrics require semantic judgment and should be manually audited even if an LLM grader is used.
- The evaluation set measures behavior near the training distribution and does not establish generalization.

## Recommended decision after the run

- **Continue** if clarification recall rises without meaningful loss of autonomy.
- **Revise the data policy** if both clarification and unnecessary-question rates rise.
- **Increase evaluation breadth before training volume** if results vary sharply by category.
- **Stop** if gains appear only in output formatting rather than underlying decision quality.

## Run 2 addition: the hard evaluation set

Run 1 saturated. Every variant, trained or not, scored 100% on the decision metrics, so the
eval could not detect a change in judgment. `data/eval_hard.jsonl` (built by
`build_hard_eval.py`) replaces it with 16 prompts written to sit close to the ASK/ACT line
and pull the model toward the wrong side:

- `ASK` cases that tempt the model to act: implied routines, recurring but unverified
  actions, requests where a plausible default exists but was never stated.
- `ACT` cases that tempt the model to over-ask: enough information to proceed, but
  open-ended framing ("draft a one-line message", "give me three hypotheses", "is 200
  examples enough?").

Same schema as `eval.jsonl`. The untrained model scores 94–96% decision accuracy and 88–92% autonomy on this set, which leaves
room for a fine-tune to move the number in either direction.

Labeling caveat: several `ACT` labels on open-ended prompts are judgment calls. A model that
asks "what data exists?" before listing hypotheses is arguably showing good instinct. The
labels reflect one consistent policy, not ground truth.

## Run 3 addition: borderline-ACT training examples

Every `ACT` example in the original training set was fully specified. Run 2 suggested the
model had learned "looks open-ended, so ask" from that. `data/train_v3.jsonl` (built by
`build_train_v3.py`) adds 12 `ACT` examples, ids `train_act_extra_01` to `_12`, that are
open-ended but answerable:

- draft-only tasks (a one-line message, an out-of-office reply)
- pre-data brainstorming (hypotheses, A/B test ideas)
- sufficient-number reasoning (is this sample size enough for a rough estimate?)
- stated-priority choices (two options, one explicit criterion)

Run 3 trains on these 52 examples for 3 epochs instead of 6. The eval is unchanged from
Run 2, so Runs 2 and 3 are directly comparable.

## Outcome against the decision rule

Run 2 and Run 3 both failed the pre-registered success criteria: clarification recall did
not rise (the base model was already at 100%), and autonomy fell by more than 5 points in
every fine-tuned variant. The only consistent gain was output formatting. The rule above
says to stop in that case, and the experiment stopped after Run 3. Numbers are in
`results/TALLY.md`.

## Changelog

- 2026-08-20, Run 1: original card, 40 training examples, 20-prompt eval.
- 2026-08-20, Run 2: added `eval_hard.jsonl`; the `contract_compliance` metric was added to the runner.
- 2026-08-20, Run 3: added `train_v3.jsonl`; epochs 6 to 3.
- 2026-10-06: card updated to describe all three runs; previously it described Run 1 only.
