# Strongest verified campaign artifact

Checkpoint: `candidate-1337-6k`. Fixed evaluation policy: context **1024**, stride **64**. Measured full-coverage BPB: **1.21114004**, over all62,021,845 prediction targets. This is independent research, not an official submission, record, or claim of global optimality.

Weights are frozen. There is no test-time training or validation-informed quantization. The script also reports the prespecified matched comparison policies; the selected policy above applies globally, never separately to individual tokens/documents.

Training launcher-to-export observed upper bound: 490.096s. Fresh evaluation command: 415.049s, including loading/compilation and all reported comparisons. Hardware: eight strict H100 SXM GPUs. Dataset is the documented decontaminated canonical SP1024 training derivative; validation is unchanged.

From this directory, in the pinned CUDA environment described in ../../REPRODUCE.md:

```bash
python ../../scripts/download_eval_inputs.py --directory .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 --max_restarts=0 --nproc_per_node=8 train_gpt.py --evaluate --directory .
```

The first command downloads only the hash-pinned canonical validation shard and tokenizer. Run evaluation offline afterward, without training data mounted. Those unchanged benchmark fixtures are supplied separately, following the upstream code-plus-weights size convention. No training data is included in this package. See ../../BEST_ARTIFACT.json for file hashes and conservative total package size, including these notices and metadata, and ../../REPORT.md for all limitations and failed approaches.
