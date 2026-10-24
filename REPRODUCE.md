# Reproduction

This is independent research, not an official submission. The scored experiment uses6,000 fixed steps after unscored throughput pilots; historical8,000-step configs are not completed results. Use local disk for the cleaned training data, with32 CPU cores and64GiB host RAM for the recorded resource configuration.

An 8×H100 SXM machine is needed for hardware/timing comparison. Costs are not zero: credits count at list price. The campaign's Modal runner requires an explicit unused `budget.py` reservation and checks STOP. Do not launch a campaign job by calling its Modal function directly.

## Data and environment
Use the Docker base `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, install GCC/G++, and the pinned Python requirements. The exact resulting package list, CUDA runtime, driver, device names and topology are saved with each completed GPU run. The local Modal client is 1.6.2.

Download the canonical published SP1024 data pinned to HF commit `a85b0e6035c3c94bc23685a07c81a8f3bf89db80`:

```bash
python -c "from scripts.prepare_data import prepare; prepare('./data')"
# Expected integrity gate: canonical training has 167 exact held-out document matches.
# After reviewing the written audit, build the fixed clean derivative:
python -c "from scripts.clean_data import clean; clean('./data')"
```

The initial prepare command intentionally raises on the documented canonical overlap after writing audit.json. The clean command removes those 167 training documents and independently checks all written documents; validation is unchanged. Both matched arms use clean_sp1024. This is not an exact published baseline reproduction.

This downloads 80 training shards (8B tokens), the entire validation shard (62,021,846 tokens, 50,000 documents), tokenizer, manifests, and performs provider-hash checks plus complete-document token-hash overlap checks. Allow at least 35GB disk for both the canonical and cleaned copies (plus staging space if used). The 48GB original source JSONL is not re-exported; the exact scope of provenance verification is recorded in the audit, including the partial trailing training document. Do not use CaseOps local rebuild defaults.

## Matched training
For each completed configuration identified in the report:

```bash
python scripts/reproduce.py --config configs/control-42-6k.json --data ./data --output ./runs/control-42
python scripts/reproduce.py --config configs/candidate-42-6k.json --data ./data --output ./runs/candidate-42
```

Seeds, fixed step count, schedules and all intentional overrides are in `configs/`. Default model/optimizer parameters are in the pinned `train_gpt.py`. All required model/training/evaluation code is bundled into `submission/train_gpt.py`, which is the executed and counted script. `vendor/train_gpt.py` is the unmodified upstream source; `scripts/instrument_baseline.py` reproduces the timing-only patch; `python scripts/build_artifact.py` deterministically bundles the frozen evaluator. Full code revision and actual source SHA256 are recorded per job. The source logs itself in the run log. `final_model.int8.ptz` is the compressed artifact; raw `final_model.pt` is diagnostic and not the evaluation artifact.

## Frozen offline evaluation
For final-phase jobs whose config has EXPORT_BEFORE_SCORING=1, use `submission/final_train_gpt.py`. Rebuild it with `python scripts/build_final_artifact.py`; this retains the identical training/quantizer/scorer and exports before held-out scoring, with redundant CPU export restricted to rank0. The recorded launcher-to-export observation includes up to10 seconds polling delay and excludes only separately timed data staging.

The final published checkpoint directory will include `config.json` and `manifest.json`. Download only the pinned validation fixtures before entering the offline environment, then evaluate:

```bash
python scripts/download_eval_inputs.py --directory artifacts/BEST_CHECKPOINT
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 --max_restarts=0 --nproc_per_node=8 submission/verified_train_gpt.py --evaluate --directory artifacts/BEST_CHECKPOINT
```

The evaluator loads weights from disk in a fresh process, checks their hash, scores validation in inference mode, compares the original baseline prefix score, and then scores the full identical target set with non-overlapping and stride64 windows. The final-phase source also includes the prespecified context2048/stride128 policy. It verifies frozen state hashes after each comparison, equal target and byte counts, and writes `evaluation.json`. `python eval_windows.py` exercises edge cases and proves exact coverage over the full target count. Causal attention is inherited from the baseline and separately tested using synthetic future-token perturbations.

Timing includes compilation and startup within the evaluator subprocess. Modal final validation also blocks external networking and mounts a separate volume containing only evaluation inputs. A full-coverage result remains an independent measurement; it is not an official leaderboard result. See the report for training timing compliance and all limitations.

The full-coverage stride1024 comparator uses a left-filled final partial window, so the last213 targets receive past context while still being scored exactly once. The original truncated-prefix score is checked separately against the training roundtrip.

The first fresh evaluator failed its strict prefix fidelity check by0.00000594 BPB. `python scripts/build_verified_evaluator.py` derives the corrected single-file evaluator from the preserved final training source. It restores upstream DDP/cache/batch execution, uses20 synthetic compilation warmups with zero optimizer steps and verified unchanged weights, and checks the literal upstream scorer plus the custom prefix against the training score before full comparisons. All real validation scoring uses inference mode; all synthetic warmup/compilation time is included. The2e-6 BPB tolerance is unchanged. See EXPERIMENTS.md for the failed attempt and exact code hashes.
