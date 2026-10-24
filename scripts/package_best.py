"""Package a measured, timing-qualified artifact; never promote an unvalidated pilot."""
import hashlib,json,os,pathlib,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
job=sys.argv[1];training=json.loads((ROOT/'results'/f'{job}.json').read_text());source=ROOT/'artifacts'/job
validation=json.loads((source/'evaluation.json').read_text());manifest=json.loads((source/'manifest.json').read_text());analysis=json.loads((ROOT/'results_summary.json').read_text())
assert training['completed_requested_steps'] and training['metrics']['launch_to_export_observed_upper_bound_seconds']<600
assert validation['under_600_seconds'] and validation['artifact_size_pass']
if job.startswith('candidate-'):assert analysis['replicated_directional_improvement'],'Do not promote an unsupported H1 change'
assert validation['frozen_state_hash_before']==validation['frozen_state_hash_after']
assert all(r['target_tokens']==62021845 and r['original_bytes']==manifest['full_validation_decoded_bytes'] for r in validation['results'][1:])
assert abs(validation['results'][0]['val_bpb']-training['metrics']['val_bpb'])<2e-6
policy=min(validation['results'][1:],key=lambda r:r['val_bpb'])
code=ROOT/'submission/final_train_gpt.py';assert hashlib.sha256(code.read_bytes()).hexdigest()==validation['source_sha256']
assert hashlib.sha256((source/'final_model.int8.ptz').read_bytes()).hexdigest()==validation['model_sha256']
out=ROOT/'submission/strongest';out.mkdir(exist_ok=True)
for name in ['final_model.int8.ptz','config.json','manifest.json']:shutil.copyfile(source/name,out/name)
shutil.copyfile(code,out/'train_gpt.py')
for name in ['LICENSE','THIRD_PARTY_NOTICES.md']:shutil.copyfile(ROOT/name,out/name)
(out/'README.md').write_text(f'''# Strongest verified campaign artifact

Checkpoint: `{job}`. Fixed evaluation policy: context **{policy['context']}**, stride **{policy['stride']}**. Measured full-coverage BPB: **{policy['val_bpb']:.8f}**, over all62,021,845 prediction targets. This is independent research, not an official submission, record, or claim of global optimality.

Weights are frozen. There is no test-time training or validation-informed quantization. The script also reports the prespecified matched comparison policies; the selected policy above applies globally, never separately to individual tokens/documents.

Training launcher-to-export observed upper bound: {training['metrics']['launch_to_export_observed_upper_bound_seconds']:.3f}s. Fresh evaluation command: {validation['subprocess_wall_seconds']:.3f}s, including loading/compilation and all reported comparisons. Hardware: eight strict H100 SXM GPUs. Dataset is the documented decontaminated canonical SP1024 training derivative; validation is unchanged.

From this directory, in the pinned CUDA environment described in ../../REPRODUCE.md:

```bash
python ../../scripts/download_eval_inputs.py --directory .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_IB_DISABLE=1 \\
  torchrun --standalone --nproc_per_node=8 train_gpt.py --evaluate --directory .
```

The first command downloads only the hash-pinned canonical validation shard and tokenizer. Run evaluation offline afterward, without training data mounted. Those unchanged benchmark fixtures are supplied separately, following the upstream code-plus-weights size convention. No training data is included in this package. See ../../BEST_ARTIFACT.json for file hashes and conservative total package size, including these notices and metadata, and ../../REPORT.md for all limitations and failed approaches.
''')
files=[{'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.is_file()]
size=sum(f['bytes'] for f in files);assert size<16000000
result={'checkpoint':job,'evidence_tier':'independent full-coverage fresh evaluation with verified standard hardware/size/training/evaluation time bounds; not official','selected_policy':policy,'training_launch_to_export_observed_upper_bound_seconds':training['metrics']['launch_to_export_observed_upper_bound_seconds'],'fresh_evaluation_command_seconds':validation['subprocess_wall_seconds'],'package_bytes_including_code_model_configuration_manifest_and_notices':size,'canonical_benchmark_fixtures_excluded_from_size':True,'files':files,'official_submission':False,'global_optimality_claim':False}
(ROOT/'BEST_ARTIFACT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files','selected_policy']},indent=2));print('BPB',policy['val_bpb'])
