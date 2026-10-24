"""Join immutable reservations, job revisions and measured results into a reviewable ledger."""
import json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
budget=json.loads((root/'budget.json').read_text());jobs=json.loads((root/'jobs.json').read_text());records=[]
records.append({'job':'controller-setup-storage-egress','conservative_charge_usd':8.,'kind':'overhead allowance, not invoice','evidence':'evidence/azure-price.json'})
for j in jobs:
 name=j['job'];p=root/'results'/f'{name}.json';r=json.loads(p.read_text()) if p.exists() else {}
 entries=[e for e in budget['entries'] if e['job']==name or e.get('original_job')==name]
 family='H1 matched QK initialization' if name.endswith('-6k') else 'setup / diagnostic'
 if j['kind'] in ['stage','evaluate']:family='H2/H3 frozen evaluation verification'
 environment=r.get('environment',{})
 record={'job':name,'hypothesis_family':family,'state':j['state'],'app_id':j.get('app_id'),'code_revision':j['code_revision'],'source_sha256':environment.get('source_sha256',r.get('source_sha256')),'configuration':j['config'],'started_utc':j['started_utc'],'finished_utc':j.get('finished_utc'),'original_error':j.get('error'),'recovery':j.get('recovery'),'provider_app_elapsed_seconds':j.get('elapsed_app_seconds'),'metrics':r.get('metrics',r.get('results')),'model_bytes':r.get('model_bytes'),'counted_artifact_bytes':r.get('artifact_bytes',r.get('total_artifact_bytes')),'artifact_size_pass':r.get('artifact_size_pass'),'completed_requested_steps':r.get('completed_requested_steps'),'environment_and_provenance':environment,'original_reservation_usd':next((e['reserved_usd'] for e in entries if e['job']==name),None),'conservative_charge_or_active_reservation_usd':round(sum(e['reserved_usd'] for e in entries),6),'expense_records':entries,'result_file':str(p.relative_to(root)) if r else None,'interpretation':'See EXPERIMENTS.md and REPORT.md; failed/incomplete jobs are not score evidence.'}
 records.append(record)
(root/'ledger.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
print(json.dumps({'records':len(records),'total_conservative_charge_or_reservation_usd':budget['total_reserved_usd'],'uncommitted_usd':budget['cap_usd']-budget['total_reserved_usd']}))
