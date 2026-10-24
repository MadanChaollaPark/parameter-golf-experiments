# Parameter Golf: controlled independent experiments

**Best verified full-validation score: 1.21114004 BPB.** Frozen context1,024/stride64 evaluation of a15.87MB packaged artifact, trained/exported in490.10 seconds and freshly evaluated in415.05 seconds on eight H100 SXM GPUs. This is post-competition research, **not an official result or leaderboard record**.

A matched QK-initialization change improved both predetermined seeds by a small mean0.00054080 BPB, below the hoped-for effect. Known sliding-window evaluation improved the primary checkpoint by0.03434337 BPB on identical full target coverage. Extending context to2,048 was worse. The sliding-window benefit repeats on the second seed; longer context is worse on both. Seed42 remains a training-timing-unverified pilot. See [STATUS.json](STATUS.json).

- [Report: results, limitations and failed approaches](REPORT.md)
- [Downloadable strongest artifact and instructions](submission/strongest/), [file hashes and size](BEST_ARTIFACT.json)
- [Exact training and evaluation reproduction](REPRODUCE.md)
- [Rules and bounded plan](PLAN.md), [prespecified experiments and decisions](EXPERIMENTS.md)
- [Causal coverage and byte-denominator audit](EVALUATION_AUDIT.md)
- [Training comparisons](results_summary.json), [frozen evaluation comparisons](evaluation_summary.json)
- [Joined experiment ledger](ledger.jsonl), [all budget reservations and adjustments](budget.json)

The pinned canonical training prefix contained167 exact complete-document matches to validation. They were removed before training; every retained document was rechecked, and all50,000 validation documents remain unchanged. This cleaned,6,000-step baseline derivative is not an exact reproduction of the published baseline. Historical tokenizer fitting, near-duplicates and raw-source re-export remain explicit provenance limitations.

Final conservative compute accounting is **$77 of the$80 cap**, including credits, failures, startup, CPU/RAM, storage, egress and an$8 overhead allowance. It is an estimate, not a provider invoice. All campaign apps are stopped and campaign volumes deleted; [completion audit](COMPLETION_AUDIT.json) and [shutdown evidence](evidence/shutdown.json) record verification. No competition submission is made. OpenAI Parameter Golf code is reused under its [MIT license](LICENSE) with [attribution](THIRD_PARTY_NOTICES.md). See the [official repository](https://github.com/openai/parameter-golf) for owner rules and public records.
