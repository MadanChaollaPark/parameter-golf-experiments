"""Explicit audited downward reconciliation; retains original reservation and appends adjustment."""
import argparse,datetime,fcntl,json,math,os,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('evidence');a=p.parse_args();e=json.loads(pathlib.Path(a.evidence).read_text())
assert e['provider_state']=='stopped' and int(e['provider_tasks'])==0
created=datetime.datetime.fromisoformat(e['provider_created_at']);stopped=datetime.datetime.fromisoformat(e['provider_stopped_at'])
seconds=(stopped-created).total_seconds()+e['extra_startup_teardown_margin_seconds']
rate=e['gpu_count']*.001097+e['cpu_limit']*.0000131+e['memory_limit_gib']*.00000222
bound=seconds*rate+e['additional_cpu_image_allowance_usd'];assert e['revised_charge_usd']>=bound
jobs=json.loads((ROOT/'jobs.json').read_text());job=next(j for j in jobs if j['job']==e['job']);assert job['state'] in ['failed','completed'] and job['app_id']==e['app_id']
with (ROOT/'.budget.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=ROOT/'budget.json';d=json.loads(path.read_text());rid='audited-reconciliation-'+e['job']
 assert not any(x['job']==rid for x in d['entries'])
 entry=next(x for x in d['entries'] if x['job']==e['job']);old=entry['reserved_usd'];new=e['revised_charge_usd'];assert math.isfinite(new) and 0<new<old
 d['entries'].append({'job':rid,'reserved_usd':new-old,'phase':entry['phase'],'kind':'explicit_audited_downward_adjustment','original_job':e['job'],'original_reservation_preserved_usd':old,'revised_conservative_charge_usd':new,'computed_bound_usd':bound,'evidence':str(pathlib.Path(a.evidence)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'details':'Provider app creation-to-stopped duration conservatively charged at full GPU/CPU/RAM limits, plus stated startup/teardown margin and CPU image allowance. Credits count. Not an invoice.'})
 d['total_reserved_usd']=round(sum(x['reserved_usd'] for x in d['entries']),6)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,path)
print(json.dumps({'job':e['job'],'provider_elapsed_plus_margin_seconds':seconds,'computed_bound_usd':bound,'charged_usd':new,'total_committed_usd':d['total_reserved_usd']},indent=2))
