# Reproduction

This is independent research, not an official submission. An 8×H100 SXM machine is needed for hardware/timing comparison. Costs are not zero: credits count at list price. The campaign's Modal runner requires an explicit unused `budget.py` reservation and checks STOP. Do not launch a campaign job by calling its Modal function directly.

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

This downloads 80 training shards (8B tokens), the entire validation shard (62,021,846 tokens, 50,000 documents), tokenizer, manifests, and performs provider-hash checks plus complete-document token-hash overlap checks. Allow ~17GB disk. The 48GB original source JSONL is not re-exported; the exact scope of provenance verification is recorded in the audit, including the partial trailing training document. Do not use CaseOps local rebuild defaults.

## Matched training
For each completed configuration identified in the report:

```bash
python scripts/reproduce.py --config configs/control-42.json --data ./data --output ./runs/control-42
python scripts/reproduce.py --config configs/candidate-42.json --data ./data --output ./runs/candidate-42
```

Seeds, fixed step count, schedules and all intentional overrides are in `configs/`. Default model/optimizer parameters are in the pinned `train_gpt.py`. All required model/training/evaluation code is bundled into `submission/train_gpt.py`, which is the executed and counted script. `vendor/train_gpt.py` is the unmodified upstream source; `scripts/instrument_baseline.py` reproduces the timing-only patch; `python scripts/build_artifact.py` deterministically bundles the frozen evaluator. Full code revision and actual source SHA256 are recorded per job. The source logs itself in the run log. `final_model.int8.ptz` is the compressed artifact; raw `final_model.pt` is diagnostic and not the evaluation artifact.

## Frozen offline evaluation
The final published checkpoint directory will include `config.json` and `manifest.json`. Add only the pinned canonical validation shard and tokenizer to that directory; no training shards. Run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_IB_DISABLE=1 \
  torchrun --standalone --nproc_per_node=8 submission/train_gpt.py --evaluate --directory artifacts/BEST_CHECKPOINT
```

The evaluator loads weights from disk in a fresh process, checks their hash, disables gradients, compares the original baseline prefix score, and then scores the full identical target set with non-overlapping and stride64 windows. It verifies frozen state hashes, equal target and byte counts, and writes `evaluation.json`. `python eval_windows.py` exercises edge cases and proves exact coverage over the full target count. Causal attention is inherited from the baseline and separately tested using synthetic future-token perturbations.

Timing includes compilation and startup within the evaluator subprocess. Modal final validation also blocks external networking and mounts a separate volume containing only evaluation inputs. A full-coverage result remains an independent measurement; it is not an official leaderboard result. See the report for training timing compliance and all limitations.
