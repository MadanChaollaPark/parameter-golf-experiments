"""Summarize frozen comparisons without mixing pilots with official results."""
import json,os,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
rows=[]
for j in json.loads((ROOT/'jobs.json').read_text()):
 if j['kind']!='evaluate' or j['state']!='completed':continue
 d=json.loads((ROOT/'results'/f"{j['job']}.json").read_text());full=d['results'][1:];t=json.loads((ROOT/'results'/f"{d['checkpoint']}.json").read_text())
 assert d['frozen_state_hash_before']==d['frozen_state_hash_after']
 assert all(x['target_tokens']==62021845 and x['original_bytes']==full[0]['original_bytes'] for x in full)
 by_policy={(x['context'],x['stride']):x for x in full}
 nonoverlap=by_policy[1024,1024];slide=by_policy[1024,64];longer=by_policy[2048,128]
 rows.append({'checkpoint':d['checkpoint'],'seed':t['config']['SEED'],'qk_gain_init':t['config']['QK_GAIN_INIT'],'model_sha256':d['model_sha256'],'source_sha256':d['source_sha256'],'target_tokens':full[0]['target_tokens'],'original_bytes':full[0]['original_bytes'],'baseline_prefix_bpb':d['results'][0]['val_bpb'],'fresh_prefix_minus_training_bpb':d['results'][0]['val_bpb']-t['metrics']['val_bpb'],'full_nonoverlap_bpb':nonoverlap['val_bpb'],'full_sliding1024_bpb':slide['val_bpb'],'full_sliding2048_bpb':longer['val_bpb'],'H2_delta_bpb':slide['val_bpb']-nonoverlap['val_bpb'],'H3_delta_bpb':longer['val_bpb']-slide['val_bpb'],'eval_command_seconds':d['subprocess_wall_seconds'],'artifact_bytes_with_config_manifest':d['artifact_bytes'],'training_timing_verified_under600':t['metrics'].get('launch_to_export_observed_upper_bound_seconds',float('inf'))<600,'official_result':False})
if len(rows)>1:
 assert len({r['seed'] for r in rows})==len(rows)
 assert len({r['qk_gain_init'] for r in rows})==1
 assert len({r['source_sha256'] for r in rows})==1
result={'measurements':rows,'H2_directionally_replicated':len(rows)>=2 and all(r['H2_delta_bpb']<0 for r in rows),'H3_directionally_replicated':len(rows)>=2 and all(r['H3_delta_bpb']<0 for r in rows),'claim_limit':'Descriptive fixed-policy comparisons across available checkpoints; no statistical-significance, hidden-test, novelty or official-record claim.'}
(ROOT/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
