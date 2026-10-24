# Prespecified experiment ledger

## H1: initial QK gain
- Hypothesis: initial normalized QK gain 5.0 rather than 1.5 improves attention selectivity in the official small-vocabulary baseline; expected 0.002–0.02 BPB gain, with a real possibility of failure.
- Mechanism/interpretability: one scalar initialization change; no changes to tokenizer, scorer, train set, batch, architecture, optimizer, compression or calibration.
- Planned matched pairs: seeds 42 and 1337, control before candidate. Each 8,000 optimizer steps ×524,288 tokens = 4,194,304,000 tokens. Same 1,200-step linear warmdown, 20 compile/warmup steps restored to identical initialization, no intermediate validation. No seed selection.
- This differs from the official 600-second baseline run, so it is a fixed-step screening experiment with the full baseline validation protocol. Final step count may be increased for a separately labeled strongest artifact if budget/timing permits; that run is not evidence of the isolated H1 effect.
- H100!:8 requested; check actual H100 names and NVLink topology. CPU hard8, memory hard32GiB, function900s, startup180s, child820s, app1080s. Reserve $11 per job before launching (1,100s at 8×0.001097 +8×0.0000131 +32×0.00000222 = $9.85; remaining margin covers scheduling/build/teardown). Function retries=0; no automatic GPU retries.
- Decision rule: only retain H1 as evidence-supported if two paired seeds improve without correctness/size regressions. Two seeds are descriptive replication, not enough for the README's significance requirement. One positive seed is provisional. Negative or conflicting results are published and control retained unless stronger evidence appears.
- Baseline scorer scores 62,021,632 targets; final 213 target tokens are truncated at 1,024-token sequence boundary. Preserve this protocol in matched runs and explicitly evaluate full tail in final validation. The first BOS is context, not a prediction target. No validation-informed quantization.

## Considered alternatives
- Leading CaseOps/TTT stack: potentially much stronger absolute BPB, but substantial code, build, long evaluation and data-audit burden reduce affordable controlled repeats. Excluded for this bounded first campaign; no claim that main baseline is competitive with leaders.
- Architecture (depth recurrence, gates): changes compute and optimizer behavior together; postpone until a measured baseline exists.
- Training efficiency (fused kernels/FlashAttention3): requires a different environment and compilation validation; postpone pending timing evidence.
- Compression (int6/GPTQ): possible higher capacity under size cap, but train-only calibration and artifact reconstruction need additional experiments. Official main's fixed weight-only int8 quantizer offers a clean matched starting point.

Run results, exact Git revision/config, data hashes, software versions, timing, size and costs are recorded in jobs.json, results/, artifacts/ and budget.json. Never interpret a planned entry as a completed experiment.
