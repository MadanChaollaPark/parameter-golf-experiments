# Frozen evaluation audit

This evaluator uses the unchanged canonical SP1024 validation shard and tokenizer. The final report must establish that all gates below passed before promoting a score. A planned check is not evidence that it passed.

## Exact target ownership and causality

There are 62,021,846 input tokens and **62,021,845 prediction targets**. The initial BOS token supplies context and has no prediction target. Index `t` denotes predicting `tokens[t+1]`.

For context length `L` and stride `S`, the first window owns `[0, min(N,L))`. Every subsequent window starts ownership where the preceding one ended, and owns at most `S` more targets. Its input starts at `max(0, end-L)`. Thus the owned intervals are consecutive, disjoint, and cover `[0,N)` exactly. The final partial interval is included once. Exhaustive short edge cases and an index proof over the actual target count are in [final-coverage-proof.json](evidence/final-coverage-proof.json).

For owned target `t`, the corresponding input position is `t-start`. The baseline's causal attention permits only positions `start..t`, strictly before target token `t+1`. RMS normalization and MLPs act separately on each token; targets enter only the cross-entropy calculation after logits. Future inputs in a padded/windowed batch cannot influence an earlier prediction. Synthetic future-token perturbation checks activate nonzero attention/MLP projections and mask later losses; final CPU staging checks both 1,024 and 2,048 context lengths.

Unowned labels and padded rows use cross-entropy's `ignore_index=-100`. Each batch mean is multiplied by its actual scored-target count. Contiguous rank partitions cover the window list exactly once; float64 all-reduces sum NLL, target count and byte count. The evaluator asserts the final target count.

The original truncated baseline prefix is checked separately against the training roundtrip BPB within 0.000002. Full-coverage comparisons include the additional 213 targets. The stride-1024 comparator left-fills the final partial window with available past context. No result should be inferred by comparing unequal target sets.

## Denominator

The exact official SP1024 formula is used for each target: token base bytes, plus a leading-space byte when applicable and the preceding token is not a boundary token. This is verified against independently decoding **all 50,000 complete canonical validation documents** on CPU before GPU evaluation. Every full-coverage policy must report the same byte denominator.

This verifies the canonical SentencePiece-decoded byte denominator used by the official baseline. It is not a new count of the original pre-normalization source JSONL. No custom tokenizer, case transformation, byte sidecar, or denominator adjustment is introduced. The canonical tokenizer's historical fitting was not independently reconstructed; see [provenance scope](evidence/tokenizer-provenance-scope.json).

## Frozen weights and isolated execution

The compressed checkpoint hash must match the staged manifest. A fresh process decompresses and strictly loads its state, disables parameter gradients, and scores under inference mode. State hashes must match before and after every comparison. There is no optimizer, gradient update, test-time training, n-gram adaptation, or validation-informed quantization/calibration in evaluation.

The GPU function mounts only the dedicated evaluation volume, asserts that the training mount is absent, and blocks outbound network access. Public validation fixtures are staged before evaluation. The source and all model reconstruction/scoring logic reside in one counted Python file. Manifest/configuration bytes are also counted conservatively; the unchanged canonical tokenizer is the standard benchmark fixture, as in the upstream baseline's code-plus-weights accounting. The optional size including that fixture is reported separately.

## Fixed comparisons and limits

The prespecified policies are full coverage with context/stride **1024/1024**, **1024/64**, and **2048/128**. Doubling stride in the longer-context policy bounds repeated input work; it is a coupled evaluation policy, not a pure context-only ablation. There is no per-token or per-document choice of whichever policy scores best.

The owner [evaluation policy](https://github.com/openai/parameter-golf#faq) permits different evaluation lengths within a separate ten-minute limit. The GPU subprocess has a stricter 470-second timeout on verified strict H100!:8, and the complete measured command includes loading and compilation. The full score remains an independent measurement, not an official leaderboard result.

The [accepted sliding-window record](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_SlidingWindowEval) motivated H2. We reimplemented ownership rather than copying its loop: a synthetic tail audit found repeated target ownership in that historical implementation. We make no claim about the numerical impact on its published record.
