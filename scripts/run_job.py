"""Reserve separately, then run one app with STOP polling and whole-app deadline."""
import argparse,datetime,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
CONTROL=pathlib.Path('/home/azureuser/parameter-golf-control')
os.environ['MODAL_CONFIG_PATH']=str(CONTROL/'private/modal.toml')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--kind',default='prepare');p.add_argument('--limit',type=int,default=2400);p.add_argument('--config');a=p.parse_args()
ledger=json.loads((ROOT/'budget.json').read_text());assert any(x['job']==a.job for x in ledger['entries']),'Missing reservation'
assert not (CONTROL/'STOP').exists() and not (CONTROL/'STOP_JOBS').exists()
from scripts.status import update
import modal
import modal_campaign as mc
jobs_path=ROOT/'jobs.json';jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else []
assert not any(j['job']==a.job for j in jobs),'No automatic retries / reuse'
job={'job':a.job,'kind':a.kind,'config':json.loads(pathlib.Path(a.config).read_text()) if a.config else {},'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'starting'}
jobs.append(job);jobs_path.write_text(json.dumps(jobs,indent=2)+'\n')
start=time.monotonic()
# Independent guardian covers image-build time as well as function polling.
guard=subprocess.Popen([sys.executable,'scripts/job_guard.py',a.job,str(a.limit)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
try:
 with modal.enable_output(),mc.app.run():
  job['app_id']=mc.app.app_id;job['state']='running';jobs_path.write_text(json.dumps(jobs,indent=2)+'\n');print('APP_ID',job['app_id'],flush=True)
  fn=getattr(mc,a.kind);call=fn.spawn(**job['config']);job['call_id']=call.object_id;jobs_path.write_text(json.dumps(jobs,indent=2)+'\n')
  while True:
   if (CONTROL/'STOP').exists() or (CONTROL/'STOP_JOBS').exists() or time.monotonic()-start>a.limit:
    call.cancel(terminate_containers=True);raise RuntimeError('STOP or whole-app elapsed limit')
   try:result=call.get(timeout=10);break
   except TimeoutError:continue
  (ROOT/'results'/f'{a.job}.json').write_text(json.dumps(result,indent=2)+'\n');job['state']='completed'
except BaseException as e:
 job['state']='failed';job['error']=type(e).__name__+': '+str(e);traceback.print_exc()
finally:
 job['elapsed_app_seconds']=time.monotonic()-start;job['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();jobs_path.write_text(json.dumps(jobs,indent=2)+'\n')
 guard.terminate();update(phase=job['state'],progress=f"{a.job}: {job['state']}",completed_job_ids=[j['app_id'] for j in jobs if j['state']=='completed'],next_step='Review results and correctness before reserving another job')
if job['state']!='completed':sys.exit(1)
