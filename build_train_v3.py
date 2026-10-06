"""
Builds data/train_v3.jsonl = your original 40 examples + NEW borderline ACT
examples that teach the missing skill: ACT even when the request looks
open-ended (draft-only tasks, pre-data brainstorming, sufficient-number
reasoning, stated-priority choices).

Why: the v2 run showed the fine-tune learned "looks ambiguous -> ASK" and
started over-asking on tasks that just needed a deliverable (draft a message,
list hypotheses). Every ACT example in the original data was fully specified,
so the model never saw "act despite surface ambiguity." These add exactly that.

Run:  python build_train_v3.py
"""

import json
import os

# NEW borderline-ACT teaching examples. Each says: produce the thing, don't ask.
EXTRA_ACT = [
    ("communication", "draft_only_looks_open",
     "Draft a quick note telling the team standup is moving to 10am.",
     "DECISION: ACT\nANSWER: Quick heads-up: standup is moving to 10:00 AM starting today. Same link and format — see you then."),

    ("communication", "draft_only_looks_open",
     "Write a one-line out-of-office saying I'm away today and back tomorrow.",
     "DECISION: ACT\nANSWER: I'm out of office today and will be back tomorrow; I'll respond to your message then."),

    ("product_design", "pre_data_brainstorm",
     "Give me three reasons users might churn after their first month, to investigate.",
     "DECISION: ACT\nANSWER: Three hypotheses to test: 1) they never hit the core 'aha' action in week one; 2) the value is real but pricing feels high once the trial ends; 3) a key workflow needs a teammate, and solo users stall. Each is measurable against activation, cancellation reasons, and single- vs multi-seat retention."),

    ("product_design", "pre_data_brainstorm",
     "Suggest three A/B tests we could run on the signup page.",
     "DECISION: ACT\nANSWER: 1) Reduce the form to email + password vs. the current full form (measure completion). 2) Add social sign-in vs. email-only (measure signup rate). 3) Move social proof above the form vs. below (measure scroll-to-submit). Each isolates one change and reads out on signup conversion."),

    ("analytics", "sufficient_numbers",
     "We had 800 visits and 200 signups. What's the signup rate?",
     "DECISION: ACT\nANSWER: 200 / 800 = 25%."),

    ("finance", "sufficient_numbers",
     "Revenue was $120,000 and costs were $90,000. What's the margin?",
     "DECISION: ACT\nANSWER: Profit is $30,000, a 25% margin ($30,000 / $120,000)."),

    ("finance", "sufficient_numbers_multi_step",
     "Budget is $80,000. We've spent $52,000 and have $16,000 committed. Are we within budget?",
     "DECISION: ACT\nANSWER: Spent plus committed is $52,000 + $16,000 = $68,000, which is under the $80,000 budget by $12,000. Yes, within budget."),

    ("software", "reasonable_default",
     "Write a Python function that returns whether a number is even.",
     "DECISION: ACT\nANSWER: def is_even(n): return n % 2 == 0"),

    ("model_selection", "priority_stated",
     "Model A: quality 92, 3.0s. Model B: quality 88, 0.4s. We need latency under 1 second. Which one?",
     "DECISION: ACT\nANSWER: Model B. It is the only one under the 1-second latency requirement (0.4s vs 3.0s); the 4-point quality difference is secondary to the stated constraint."),

    ("communication", "draft_only_looks_open",
     "Draft a two-sentence thank-you reply to someone who gave me detailed feedback.",
     "DECISION: ACT\nANSWER: Thank you so much for the detailed feedback — it's genuinely useful and I appreciate the time you took. I'll work through your points and follow up with what I change."),

    ("research", "reasoned_general_answer",
     "Roughly, is 30 data points enough to estimate an average reliably?",
     "DECISION: ACT\nANSWER: Roughly: 30 is the common rule-of-thumb minimum for a stable mean, but the confidence interval is still wide (it shrinks with the square root of n), so treat it as a ballpark, not a precise figure — and it's less reliable if the data is skewed or has outliers."),

    ("scheduling", "sufficient_numbers",
     "A meeting is 90 minutes and starts at 2:00 PM. When does it end?",
     "DECISION: ACT\nANSWER: 3:30 PM."),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    orig = os.path.join(here, "data", "train_original.jsonl")
    out = os.path.join(here, "data", "train_v3.jsonl")

    rows = []
    with open(orig, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    start = len(rows)
    for i, (cat, amb, user, assistant) in enumerate(EXTRA_ACT, 1):
        rows.append({
            "id": f"train_act_extra_{i:02d}", "split": "train", "category": cat,
            "label": "ACT", "ambiguity_type": amb,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ask = sum(1 for r in rows if r["label"] == "ASK")
    n_act = sum(1 for r in rows if r["label"] == "ACT")
    print(f"Wrote {len(rows)} examples ({n_ask} ASK / {n_act} ACT) -> {out}")
    print(f"Added {len(rows)-start} new borderline-ACT examples.")


if __name__ == "__main__":
    main()
