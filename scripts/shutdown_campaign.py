"""Finalize only this campaign's Modal resources after verified local artifact preservation."""
import datetime,hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['MODAL_CONFIG_PATH']='/home/azureuser/parameter-golf-control/private/modal.toml'
import modal
python='/home/azureuser/parameter-golf-control/venv/bin/python'
assert (ROOT/'BEST_ARTIFACT.json').exists(),'Package and verify the deliverable before normal shutdown'
best=json.loads((ROOT/'BEST_ARTIFACT.json').read_text())
for f in best['files']:
 p=ROOT/f['file'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
jobs=json.loads((ROOT/'jobs.json').read_text());assert not any(j['state'] in ['running','starting'] for j in jobs)
for j in jobs:
 if j['kind']=='train' and j['state']=='completed':
  d=json.loads((ROOT/'results'/f"{j['job']}.json").read_text());f=d['artifacts'][0];p=ROOT/'artifacts'/j['job']/'final_model.int8.ptz';assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 if j['kind']=='evaluate' and j['state']=='completed':
  assert (ROOT/'artifacts'/j['config']['job_id']/'evaluation_download_manifest.json').exists()
ids={j['app_id'] for j in jobs if j.get('app_id')}
def app_state():
 rows=json.loads(subprocess.check_output([python,'-m','modal','app','list','--json'],text=True));rows=[r for r in rows if r['app_id'] in ids]
 assert {r['app_id'] for r in rows}==ids
 assert all(r['description']=='parameter-golf-controlled-20261024' for r in rows)
 return rows
before=app_state()
for r in before:
 if r['state']!='stopped' or int(r['tasks'])!=0:subprocess.run([python,'-m','modal','app','stop',r['app_id']],check=True,capture_output=True,timeout=45)
after=app_state();assert all(r['state']=='stopped' and int(r['tasks'])==0 for r in after)
volumes=['parameter-golf-controlled-20261024','parameter-golf-evaluation-20261024'];claims=[]
for name in volumes:
 v=modal.Volume.from_name(name)
 for f in v.listdir('execution_claims'):
  data=b''.join(v.read_file(f.path));claims.append({'volume':name,'path':f.path,'claim':json.loads(data)})
(ROOT/'evidence/execution-claims.json').write_text(json.dumps(claims,indent=2)+'\n')
for name in volumes:modal.Volume.objects.delete(name,allow_missing=True)
listed=json.loads(subprocess.check_output([python,'-m','modal','volume','list','--json'],text=True))
assert not any(r.get('name',r.get('Name')) in volumes for r in listed)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_recorded_apps_stopped':True,'active_tasks':0,'provider_apps':after,'deleted_campaign_volumes':volumes,'volume_absence_verified':True,'compressed_models_and_evaluation_logs_preserved_locally':True,'production_azure_vm_and_other_services_preserved':True,'watchdog_service_preserved':'parameter-golf-watchdog.service','FINALIZED_marker_still_requires_publication':True}
(ROOT/'evidence/shutdown.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='provider_apps'},indent=2))
