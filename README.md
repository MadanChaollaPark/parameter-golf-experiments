# Parameter Golf independent experiments

Controlled, post-competition research under an $80 compute budget. **Campaign in progress; no official result or competitive-win claim.**

The first complete fixed-step control measured **1.24724648 BPB**, with **15,859,511 bytes** of code plus compressed weights. Its combined training/validation/export timing exceeds 600 seconds, so it remains a pilot with unverified standard-track timing. The matched QK-initialization candidate is running; final offline verification is pending.

The pinned canonical training prefix contained 167 exact full-document matches to validation. They were removed before training, and every retained training document was rechecked. Validation remains unchanged. This cleaned derivative is an explicit difference from the published baseline; tokenizer provenance and near-duplicate limits are documented.

- [Plan and rules audit](PLAN.md)
- [Prespecified hypotheses, changes and failures](EXPERIMENTS.md)
- [Reproduction](REPRODUCE.md)
- [Durable status](STATUS.json), [joined experiment ledger](ledger.jsonl), [conservative budget](budget.json)
- [Original repository](https://github.com/openai/parameter-golf)

This repository reuses OpenAI Parameter Golf under its MIT license and preserves [attribution](THIRD_PARTY_NOTICES.md). No competition submission will be made.
