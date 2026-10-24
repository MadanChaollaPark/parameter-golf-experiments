"""Compare only completed matched fixed-step runs; never promote a failed pilot."""
import json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
runs={};pairs=[]
for arm in ['control','candidate']:
 for seed in [42,1337]:
  name=f'{arm}-{seed}-6k';path=ROOT/'results'/f'{name}.json'
  if not path.exists():continue
  d=json.loads(path.read_text());runs[name]=d
for seed in [42,1337]:
 c=runs.get(f'control-{seed}-6k');t=runs.get(f'candidate-{seed}-6k')
 if not c or not t:continue
 assert c['completed_requested_steps'] and t['completed_requested_steps'],'Truncated runs are not matched evidence'
 assert c['artifact_size_pass'] and t['artifact_size_pass']
 cc=dict(c['config']);tc=dict(t['config']);cq=cc.pop('QK_GAIN_INIT');tq=tc.pop('QK_GAIN_INIT')
 assert cc==tc and cq=='1.5' and tq=='5.0'
 for key in ['source_sha256','clean_audit_sha256','data_revision','torch','cuda','gpu_names']:
  assert c['environment'][key]==t['environment'][key],f'Mismatch: {key}'
 assert c['baseline_eval_targets']==t['baseline_eval_targets']
 pairs.append({'seed':seed,'control_bpb':c['metrics']['val_bpb'],'candidate_bpb':t['metrics']['val_bpb'],'candidate_minus_control_bpb':t['metrics']['val_bpb']-c['metrics']['val_bpb'],'matched_source_sha256':c['environment']['source_sha256'],'matched_clean_data_sha256':c['environment']['clean_audit_sha256']})
if len(pairs)==2:
 supported=all(x['candidate_minus_control_bpb']<0 for x in pairs)
 decision='Retain QK5 as descriptively replicated improvement; no significance or official-win claim' if supported else 'No consistent two-seed improvement; retain control for primary conclusion'
elif len(pairs)==1:
 supported=False;decision='One matched seed only: provisional, not a replicated improvement'
else:supported=False;decision='No completed matched pair yet'
summary={'pairs':pairs,'replicated_directional_improvement':supported,'decision':decision,'scored_runs':{name:{'bpb':d['metrics']['val_bpb'],'model_bytes':d['model_bytes'],'total_counted_training_artifact_bytes':d['total_artifact_bytes'],'completed_requested_steps':d['completed_requested_steps'],'main_to_export_seconds':d['metrics']['campaign_to_export_end_seconds'],'subprocess_wall_seconds':d['subprocess_wall_seconds'],'staging_seconds':d['staging']['seconds']} for name,d in runs.items()}}
if pairs:summary['mean_paired_delta_bpb']=statistics.mean(p['candidate_minus_control_bpb'] for p in pairs)
if len(pairs)>1:summary['paired_delta_sample_std_bpb']=statistics.stdev(p['candidate_minus_control_bpb'] for p in pairs)
(ROOT/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
