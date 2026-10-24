# Controlled Parameter Golf campaign

The strongest verified artifact scores **1.21114004 bits per byte (BPB)** on the complete canonical validation target set. It uses frozen weights with context1,024 and stride64. This is independent post-competition research, **not an official submission or leaderboard result**. The artifact, primary verification and prespecified second-seed frozen-policy check are complete. Publication and provider shutdown are verified separately in COMPLETION_AUDIT.json and evidence/shutdown.json.

The campaign found a small, directionally consistent QK-initialization improvement and a much larger benefit from the already-known sliding-window evaluation method. Longer context was harmful. It did **not** reach the public leading scores around1.0565 BPB, demonstrate statistical significance, or establish that this is the strongest possible configuration under every allocation of the budget.

## Primary artifact and measured constraints

| Item | Measurement |
|---|---:|
| Checkpoint | candidate-1337-6k |
| Full-coverage BPB, context1024/stride64 | 1.2111400440145654 |
| Prediction targets | 62,021,845 |
| Canonical decoded-byte denominator | 151,080,891 |
| Compressed weights | 15,804,351 bytes |
| Entire packaged deliverable, including code/config/manifest/notices | 15,868,536 bytes |
| Training launch through export, conservative observed upper bound | 490.096133 seconds |
| Fresh evaluation subprocess, including loading/compilation and every comparison | 415.048886 seconds |
| Hardware | 8 same-host NVIDIA H100 SXM 80GB |

The deliverable is [submission/strongest](submission/strongest/); [BEST_ARTIFACT.json](BEST_ARTIFACT.json) records every file's SHA256 and size. The training/export timer includes launcher/imports, compilation, warmup, all updates, raw serialization, compression and reload, plus up to10 seconds observation delay. Data staging is separately measured and fully charged. Fresh evaluation includes20 synthetic compilation warmups, the literal upstream reference, and all four scored comparisons. Both measured commands meet their separate600-second limits. As an additional conservative check, the primary training app's entire provider creation-to-stop lifetime was564 seconds, including staging, training, export and its post-export scoring. No time is inferred by subtracting an unmeasured validation interval.

The selected model SHA256 is `70d2b520c406ef459314e2540c0d8087aefcc3a565cf000e1647686209e7bffa`. Evaluation source SHA256 is `ae7430a9f0f3de13f0eb264394cb039ef39c81e7a784bbabde0d4d377a30bff4`. Frozen reconstructed-state hashes match before and after every comparison and synthetic warmup. The complete package fits the decimal16,000,000-byte limit under upstream's code-plus-weights convention: canonical tokenizer and validation data are separately supplied benchmark fixtures. Including the254,483-byte tokenizer would exceed16MB; this exclusion is explicit, not a hidden sidecar.

## H1: matched training experiment

Hypothesis: change only normalized attention QK gain initialization from1.5 to5.0, expecting0.002–0.02 BPB improvement at the same training budget. Each pair used the same seed, cleaned data, model, optimizer, batch,6,000 steps, schedule, quantizer, source and hardware configuration. The global batch is524,288 tokens, giving3,145,728,000 trained tokens per run. Training uses the official baseline's9-layer,512-wide,17,059,912-parameter tied-embedding SP1024 model and Muon optimizer.

| Seed | QK1.5 control BPB | QK5 candidate BPB | Candidate minus control | Training timing status |
|---|---:|---:|---:|---|
| 42 | 1.24724648 | 1.24678452 | -0.00046196 | Both pilot timing unverified |
| 1337 | 1.24610386 | 1.24548422 | -0.00061964 | Both launch-to-export under600s |

These scores use the original baseline's62,021,632-target truncated coverage. The seed1337 control provides the bounded timing-qualified baseline:1.24610386 BPB,540.098856 seconds launch-to-export,15,803,821 weight bytes plus56,359 source bytes, and an$8 conservative compute charge. It is not the full13,780-step published baseline. Mean paired delta is **-0.00054080 BPB**; paired-delta sample SD is0.00011150. Both seeds improve, so QK5 is retained with a narrow descriptive claim. The gain is below the hypothesized minimum. Two seeds do not establish statistical significance. Seed1337 is the timing-qualified primary artifact; seed42 remains a pilot even if fresh evaluation succeeds.

Between seed pairs, export was moved before the unquantized held-out diagnostic, and identical CPU quantization/compression was restricted to rank0. Both arms within each pair share the same code. AST checks preserve model, optimizer, quantizer and scorer definitions. These coupled post-training changes allow direct timing, without attributing runtime differences to QK gain. Host/compiler variability also prevents a causal throughput claim. [results_summary.json](results_summary.json) and each run's environment, source hash, config and logs preserve the evidence.

## H2/H3: frozen, matched-coverage evaluation

Before any sliding-window scores, fix three policies:1024/1024,1024/64 and2048/128. H2 expected0.02–0.04 BPB benefit from preserving past context at most target positions. H3 expected0–0.01 additional improvement, with possible degradation because training context is1024. H3 doubles stride together with context to bound work; it is a coupled policy comparison, not a pure context-only ablation. No further context or stride sweep was performed.

| Primary seed1337 policy | Full BPB | Difference from preceding policy | Decision |
|---|---:|---:|---|
| Context1024, stride1024 | 1.245483411841 | — | Matched full-coverage control |
| Context1024, stride64 | 1.211140044015 | -0.034343367826 | Retain H2 |
| Context2048, stride128 | 1.251430533318 | +0.040290489303 | Reject H3 |

Every row scores exactly the same62,021,845 targets and151,080,891 bytes. The extra213 tail targets omitted by native baseline scoring are included once; the full stride1024 comparator left-fills its final partial window. The original prefix is separately verified. The selected policy applies globally, never by taking the best loss per token/document. The primary fresh prefix agrees with native scoring within1.8e-9 BPB, much tighter than the unchanged2e-6 gate.

The second-seed check passed with identical source and policies, using the existing QK5 checkpoint and no new training:

| Seed42 policy | Full BPB | Difference from preceding policy |
|---|---:|---:|
| Context1024, stride1024 | 1.246783794996 | — |
| Context1024, stride64 | 1.212425187122 | -0.034358607874 |
| Context2048, stride128 | 1.259088048989 | +0.046662861868 |

Its fresh subprocess took405.057711 seconds, and native prefix agreement was within7.7e-10 BPB. Exact coverage, byte denominator, frozen weights, isolated inputs and strict hardware all pass. H2 improves both seeds by about0.03435 BPB, and H3 worsens both. This strengthens the fixed-policy finding without establishing formal significance. Seed42's original training timing remains unverified; it is a full-evaluation replication on a pilot checkpoint, not a second standard-timing artifact. [evaluation_summary.json](evaluation_summary.json) contains both complete comparisons.

## Data, causal correctness and permitted evaluation

Upstream was freshly checked at `f5c079314c4877fbb0af378c0abade5a8ca33d3a`. Owner rules allow run-level validation tuning, different evaluation context lengths and a separate ten-minute evaluation budget. The challenge dates have passed. The accepted [SlidingWindowEval record](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_SlidingWindowEval) motivated H2; this work does not claim that idea is novel. We use no test-time training, parameter adaptation or validation-informed quantization/calibration.

All80 canonical training shards and the full validation shard/tokenizer were pinned to Hugging Face revision `a85b0e6035c3c94bc23685a07c81a8f3bf89db80` and checked against provider hashes. The audit found167 exact complete tokenized-document training/validation overlaps in the8B-token prefix. Before any training, the fixed cleaning pass removed all167 documents (205,970 tokens), dropped the unfinished1,011-token trailing training document, and re-read all6,292,772 retained training documents to establish zero exact full-document matches. The retained derivative contains7,999,793,019 tokens. Validation remains all50,000 canonical documents, including its two duplicate-content documents. Hash comparison serves only the required leakage exclusion; no held-out tokens are used to fit weights or calibrate quantization.

This is a **cleaned, shorter-training derivative of the official baseline**, not an exact reproduction of its published1.22436570 score/13,780-step run. The source manifest and full tokenized-document split were audited, but the48GB raw source JSONL was not independently re-exported, near-duplicates were not excluded, and historical canonical tokenizer fitting was not independently reconstructed. Current export helper code is not evidence of the historical tokenizer's fitting split. These provenance limits prevent a stronger clean-source claim.

The official SentencePiece byte formula was independently checked against decoding every validation document: both total151,080,891 bytes. This is the official canonical decoded-byte convention, not a reconstruction of pre-normalization source JSONL bytes. Coverage intervals are consecutive and disjoint; index proofs and edge cases establish every target is scored once. Causal attention sees only positions preceding its target. Nonzero synthetic-model future-token perturbation checks pass at1024 and2048 contexts. See [EVALUATION_AUDIT.md](EVALUATION_AUDIT.md) for the proof and denominator details.

Fresh GPU evaluation mounts only an evaluation volume, asserts training data is absent, and blocks external networking. All real validation scoring is inference-only. The20 synthetic forward/backward warmups use zeros, no optimizer and zero weight updates; state hashes verify that they only reproduce compilation/cache behavior. They are included in evaluation timing and are not test-time training.

Strict Modal `H100!:8` was checked in the SDK and against actual GPU names. All56 directed NVLink peer connections and all18 links/device pass, as does CUDA peer access. The broad `nvidia-smi topo -m` CPU/NUMA matrix is unavailable in the container; this limitation is preserved. Recorded environment: PyTorch2.8.0+cu128, CUDA12.8, driver610.57.04, cuDNN91002; per-job files retain full versions.

## Considered directions

Architecture changes (depth/width/attention structure) were deferred because they would require another compiled training control and introduce several coupled capacity/throughput changes. Optimization work was restricted to the single QK initialization scalar. Training efficiency work staged verified bytes locally and avoided redundant export; its isolated speed benefit was not established. Compression retained the official weight-only int8/zlib quantizer, whose AST and reconstruction checks pass; more aggressive compression was unnecessary for the measured artifact size and would require additional matched quality verification. These are considered but untested alternatives, not negative experimental results.

The largest affordable remaining opportunity after H1 was a permitted evaluation-policy change on existing weights; H2 demonstrated a substantially larger effect without another training run. Budget was then spent on the prespecified independent seed check rather than an unbounded architecture/context sweep.

## Failures and discarded approaches

| Attempt | Finding and disposition |
|---|---|
| First CPU image build | Source excluded from builder; failed before GPU allocation. Corrected explicitly and charged. |
| Canonical data gate | Detected167 exact overlaps; no training before fixed cleaning and re-audit. |
| First GPU hardware gate | Broad topology CLI failed before optimization. Separate paid diagnostic established NVLink connectivity; failure retained. |
| 8,000-step direct-volume pilot | ~82ms/step plus setup projected beyond limit. Cancelled, no score/artifact claim. |
| 8,000-step staged-data pilot | ~72ms/step still projected too long. Cancelled before initial unsynchronized guard; subsequent guard corrected. No score. |
| Seed42 timing protocol | Export followed an unquantized diagnostic; conservative training compliance cannot be proved. Preserve as pilot. |
| Control42 result transport | TorchVersion deserialization failed locally after remote completion. Recovered persisted JSON; no GPU rerun. |
| First fresh evaluation | Prefix differed by5.94e-6 BPB, failing2e-6 gate. All scores withheld; no tolerance relaxation. |
| Corrected fresh protocol | Restored upstream DDP/cache/warmup and actual batch sizes together. Literal reference and custom prefix pass. Individual cause of earlier numerical discrepancy was not isolated. Separate explicit reservation/claim, no automatic retry. |
| Longer-context H3 | Worse than1024/64 by0.04029 BPB on primary; reject. |

The older accepted sliding-window loop has duplicate target ownership in a synthetic tail case. We implemented explicit ownership instead of copying it and make no claim about its published score's numerical impact. The complete chronological hypotheses and failed decisions remain in [EXPERIMENTS.md](EXPERIMENTS.md); expense records are never deleted.

## Budget, reproducibility and completion

The hard cap is$80 at full list cost, including credits. Every billable build/job had a prior reservation; final verification had a protected reserve. Original expenses remain in [budget.json](budget.json), with explicit evidence-backed adjustments only after provider shutdown. Reconciliations conservatively charge the entire app lifetime at maximum GPU/CPU/RAM allocation, add startup/teardown margin and image allowance, and round upward. They are **estimates, not a provider invoice**. The initial$8 Azure/controller/storage/egress allowance remains charged. Azure retail Standard_D4s_v4 Belgium Central Linux was verified at$0.23/hour; eight hours=$1.84 fits that allowance.

The final conservative charge is **$77.00**, leaving **$3.00 unspent/unreserved** under the cap. This includes every failed attempt and the entire$8 overhead allowance. No further compute job is planned.

| Cost category | Conservative charge |
|---|---:|
| Azure/controller/storage/egress allowance | $8.00 |
| Image/data preparation, cleaning and two CPU evaluation stages | $2.50 |
| Hardware gate/diagnostic and two cancelled throughput pilots | $19.00 |
| Four completed matched training runs | $32.00 |
| Rejected fresh evaluator and two verified fresh evaluations | $15.50 |
| **Total** | **$77.00** |

[Shutdown evidence](evidence/shutdown.json) verifies every recorded campaign app stopped with zero tasks and both campaign volumes deleted after local artifact preservation. The existing production Azure VM, unrelated resources and watchdog service are preserved. [Completion audit](COMPLETION_AUDIT.json) records unauthenticated public repository verification and the audited deliverables commit. No automated GPU research retry is enabled: ordinary Functions have retries0/max_containers1, hard limits and task timeouts; persistent execution claims fail closed on infrastructure restarts. STOP and whole-app deadlines are monitored independently. No other Azure services or datasets are used.

[REPRODUCE.md](REPRODUCE.md) gives exact training and offline evaluation commands. [ledger.jsonl](ledger.jsonl) joins hypotheses, Git revisions, configs/seeds, source/data/environment hashes, metrics, errors and charges. Reused OpenAI code retains its MIT license and [attribution](THIRD_PARTY_NOTICES.md). Public leading submissions were inspected, including PR2135/PR2014 and clean-data audits. Their reported p-values and README threshold differ; no official winner or significance claim is inferred. This campaign deliberately used the auditable main baseline within its budget rather than implementing a multi-component leading TTT stack.

## Outcome

The bounded campaign produced a reproducible, size- and timing-qualified independent artifact, matched evidence across two training seeds, a repeated sliding-window benefit, a documented negative longer-context result, and a public audit trail within the$80 cap. The competitive leading-score target was **not achieved**; no official submission, record, significance claim or global-optimality claim is made. The strongest supported artifact from this campaign is1.21114004 BPB.

## Authorized continuation

After the original campaign completed, the human clarified that the intended starting point was an actual leading implementation and authorized up to$20 additional compute. The original shortened starter baseline did not satisfy that intended research direction. Its results and$77 charge remain preserved. The separate [leading-baseline extension report](extension/REPORT.md) and [plan](extension/PLAN.md) document the correction, separate costs, measured outcome and limits.
