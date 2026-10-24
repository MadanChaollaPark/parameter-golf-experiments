"""Restore the upstream DDP/cache/batch execution protocol; keep the strict fidelity gate."""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parents[1];old=(root/'submission/final_train_gpt.py').read_text()
assert hashlib.sha256(old.encode()).hexdigest()=='a9f05de2296f313cf7daacb4366a7067eb494fda3116030fcd7ca91b365be42e'
s=old.replace('x=torch.zeros((batch_size,context),dtype=torch.int64)','x=torch.zeros((len(batch),context),dtype=torch.int64)')
needle=" for param in model.parameters():param.requires_grad_(False)\n before=state_hash(model);compiled=torch.compile(model,dynamic=False,fullgraph=True)"
replacement=''' # Match upstream compilation/position-cache/DDP execution. Synthetic gradient
 # warmup primes kernels and DDP buckets; NO optimizer or weight update occurs.
 before=state_hash(model)
 from torch._dynamo import config as dynamo_config
 dynamo_config.cache_size_limit=32  # Explicitly cover the prespecified batch/context shapes.
 compiled_base=torch.compile(model,dynamic=False,fullgraph=True)
 compiled=DDP(compiled_base,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
 warm_start=time.perf_counter()
 synthetic_x=torch.zeros((64,1024),dtype=torch.int64,device=device);synthetic_y=torch.zeros_like(synthetic_x)
 compiled.train()
 for _ in range(20):
  with torch.autocast('cuda',dtype=torch.bfloat16):synthetic_loss=compiled(synthetic_x,synthetic_y)
  synthetic_loss.backward();compiled.zero_grad(set_to_none=True)
 del synthetic_loss,synthetic_x,synthetic_y
 torch.cuda.synchronize()
 assert before==state_hash(model),'Synthetic compilation warmup changed weights'
 warmup={'synthetic_only':True,'iterations':20,'optimizer_steps':0,'weights_unchanged':True,'seconds':time.perf_counter()-warm_start}
'''
assert needle in s;s=s.replace(needle,replacement.rstrip())
needle=' # Baseline prefix fidelity check, then matched full-coverage H2 comparison.'
replacement=''' # Require the literal upstream scorer to reproduce the training roundtrip
 # before spending on any window comparisons; do not relax the existing tolerance.
 reference_start=time.perf_counter()
 reference_loss,reference_bpb=m.eval_val(args,compiled,rank,world,device,1,tokens[:n//1024*1024+1],base.to(device),leading.to(device),boundary.to(device))
 reference={'val_loss':reference_loss,'val_bpb':reference_bpb,'expected_training_bpb':manifest['expected_baseline_bpb'],'difference_bpb':reference_bpb-manifest['expected_baseline_bpb'],'seconds_including_compile':time.perf_counter()-reference_start}
 if rank==0:(root/'reference_fidelity.json').write_text(json.dumps({'reference':reference,'synthetic_warmup':warmup,'weights_unchanged':before==state_hash(model)},indent=2))
 assert abs(reference['difference_bpb'])<2e-6,'Literal upstream fresh scorer disagrees with training roundtrip'
'''+needle
assert needle in s;s=s.replace(needle,replacement)
needle="  checkpoint_hash=state_hash(model);assert before==checkpoint_hash"
s=s.replace(needle,needle+"\n  if len(results)==1:\n   assert abs(results[0]['val_bpb']-reference_bpb)<2e-6,'Window prefix differs from literal upstream scorer'\n   assert abs(results[0]['val_bpb']-manifest['expected_baseline_bpb'])<2e-6,'Fresh window prefix differs from training roundtrip'")
s=s.replace("'completed_comparisons':len(results),'no_ttt':True", "'completed_comparisons':len(results),'reference_prefix':reference,'synthetic_warmup':warmup,'no_ttt':True")
s=s.replace("'total_script_seconds':time.perf_counter()-SCRIPT_START,'torch':torch.__version__", "'total_script_seconds':time.perf_counter()-SCRIPT_START,'reference_prefix':reference,'synthetic_warmup':warmup,'torch':torch.__version__")
p=root/'submission/verified_train_gpt.py';p.write_text(s)
a={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
b={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
unchanged=sorted(set(a)-{'score','evaluation_main'});assert all(a[k]==b[k] for k in unchanged)
e={'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'source_bytes':len(s.encode()),'unchanged_definitions':unchanged,'model_training_quantizer_original_scorer_window_ownership_AST_identical':True,'fidelity_tolerance_bpb_unchanged':.000002,'change':'Restore upstream DDP, synthetic compiler/position-cache warmup with zero optimizer steps and actual final batch size. Literal upstream scorer and custom prefix must both match native training score before full comparisons. All warmup/compile time counted. No validation gradients or weight updates.'}
(root/'evidence/verified-evaluator-source.json').write_text(json.dumps(e,indent=2)+'\n');print(json.dumps(e,indent=2))
