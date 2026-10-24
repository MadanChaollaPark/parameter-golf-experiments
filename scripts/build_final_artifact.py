"""Derive final-phase source: export frozen weights before any held-out scoring.

No learned computation, initialization, training step, quantizer or scorer changes.
Keep submission/train_gpt.py intact for the first matched pair.
"""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parents[1]
s=(root/'submission/train_gpt.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='02099aa4623012c9bbe020215e6579ea793c00deb99e28fcea93a2c33f696a67'
needle='        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)'
replacement='''        # Final verification mode: no held-out scoring before the serialized artifact exists.
        if last_step and os.environ.get("EXPORT_BEFORE_SCORING") == "1":
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            log0(f"campaign_to_train_end_seconds:{time.perf_counter()-campaign_start:.6f}")
            log0(f"step:{step}/{args.iterations} training_complete train_time:{training_time_ms:.0f}ms")
            break

'''+needle
assert s.count(needle)==1;s=s.replace(needle,replacement)
needle='    log0(f"campaign_to_export_end_seconds:{time.perf_counter()-campaign_start:.6f}")'
s=s.replace(needle,needle+'\n    log0(f"campaign_script_to_export_end_seconds:{time.perf_counter()-SCRIPT_START:.6f}")')
assert 'campaign_script_to_export_end_seconds' in s
# Every DDP rank currently repeats identical CPU quantization/compression, although
# only rank0 writes. Keep rank0's exact operations and let peers wait at the existing barrier.
start=s.index('    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())')
end=s.index('    if master_process:',start)
block=s[start:end]
assert 'dist.' not in block and 'quant_raw_bytes = len(quant_raw)' in block
s=s[:start]+'    if master_process:\n'+''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True))+s[end:]
# One prespecified longer-context comparison, with stride doubled to bound work.
s=s.replace('device,base,leading,boundary):\n spec=list(windows(n_targets,1024,stride))', 'device,base,leading,boundary,context=1024):\n spec=list(windows(n_targets,context,stride))')
s=s.replace('x=torch.zeros((batch_size,1024),dtype=torch.int64)', 'x=torch.zeros((batch_size,context),dtype=torch.int64)')
s=s.replace("'context':1024,'nll_sum'", "'context':context,'nll_sum'").replace('prove_coverage(n_targets,1024,stride)', 'prove_coverage(n_targets,context,stride)')
s=s.replace("p.add_argument('--directory',required=True);a=p.parse_args()", "p.add_argument('--directory',required=True);p.add_argument('--skip-long-context',action='store_true');a=p.parse_args()")
old=" results=[score(compiled,tokens,n//1024*1024,1024,rank,world,device,base,leading,boundary),score(compiled,tokens,n,1024,rank,world,device,base,leading,boundary),score(compiled,tokens,n,64,rank,world,device,base,leading,boundary)]"
new=""" results=[]
 specifications=[(n//1024*1024,1024,1024),(n,1024,1024),(n,64,1024)]
 if not a.skip_long_context:specifications.append((n,128,2048))
 for target_count,stride,context in specifications:
  results.append(score(compiled,tokens,target_count,stride,rank,world,device,base,leading,boundary,context))
  checkpoint_hash=state_hash(model);assert before==checkpoint_hash
  if rank==0:
   (root/'partial_evaluation.json').write_text(json.dumps({'checkpoint':root.name,'model_sha256':blob_sha,'frozen_state_hash_before':before,'frozen_state_hash_after':checkpoint_hash,'results':results,'total_script_seconds':time.perf_counter()-SCRIPT_START,'completed_comparisons':len(results),'no_ttt':True},indent=2))
"""
assert old in s;s=s.replace(old,new.rstrip())
s=s.replace(" assert results[1]['target_tokens']==results[2]['target_tokens'] and results[1]['original_bytes']==results[2]['original_bytes']", " assert all(r['target_tokens']==results[1]['target_tokens'] and r['original_bytes']==results[1]['original_bytes'] for r in results[2:])")
(root/'submission/final_train_gpt.py').write_text(s)
a={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/'vendor/train_gpt.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
b={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
keys=sorted(set(a)-{'main'});assert all(a[k]==b[k] for k in keys)
e={'original_model_optimizer_quantizer_and_baseline_evaluator_AST_identical':True,'compared_definitions':keys,'bundled_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'bundled_source_bytes':len(s.encode()),'change':'Rank0-only identical CPU export; prespecified longer-context2048/stride128 frozen comparison added. Optional EXPORT_BEFORE_SCORING=1 skips unquantized held-out diagnostic after final optimizer step; export and reconstruction precede all held-out scoring. Extra timer begins before Python imports. Same compressed roundtrip scorer.'}
(root/'evidence/code-equivalence-final.json').write_text(json.dumps(e,indent=2)+'\n');print(json.dumps(e,indent=2))
