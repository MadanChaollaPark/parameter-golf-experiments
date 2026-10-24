"""Read-only provider evidence and conservative proposed reconciliation; does not alter expenses."""
import argparse,datetime,json,math,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['MODAL_CONFIG_PATH']='/home/azureuser/parameter-golf-control/private/modal.toml'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--gpu',type=int,default=8);p.add_argument('--cpu',type=int,default=32);p.add_argument('--memory',type=int,default=64);p.add_argument('--margin',type=int,default=120);a=p.parse_args()
j=next(j for j in json.loads((ROOT/'jobs.json').read_text()) if j['job']==a.job);assert j['state'] in ['completed','failed']
state=json.loads(subprocess.check_output(['/home/azureuser/parameter-golf-control/venv/bin/python','-m','modal','app','list','--json'],text=True))
owned=[r for r in state if r['description']=='parameter-golf-controlled-20261024'];r=next(r for r in owned if r['app_id']==j['app_id']);assert r['state']=='stopped' and int(r['tasks'])==0
snapshot='evidence/provider-state-after-'+a.job+'.json';(ROOT/snapshot).write_text(json.dumps(owned,indent=2)+'\n')
seconds=(datetime.datetime.fromisoformat(r['stopped_at'])-datetime.datetime.fromisoformat(r['created_at'])).total_seconds()+a.margin
bound=seconds*(a.gpu*.001097+a.cpu*.0000131+a.memory*.00000222)+.25
charge=math.ceil(bound*2)/2
result={'job':a.job,'app_id':r['app_id'],'provider_state':r['state'],'provider_tasks':r['tasks'],'provider_created_at':r['created_at'],'provider_stopped_at':r['stopped_at'],'extra_startup_teardown_margin_seconds':a.margin,'gpu_count':a.gpu,'cpu_limit':a.cpu,'memory_limit_gib':a.memory,'additional_cpu_image_allowance_usd':.25,'revised_charge_usd':charge,'evidence_source':snapshot+f'; full provider app lifetime charged at simultaneous maximum resource limits, including startup/staging/compilation/export/evaluation; additional{a.margin}s termination uncertainty margin for completed jobs plus cached image allowance; rounded upward to$0.50'}
target='evidence/'+a.job+'-cost-reconciliation.json';(ROOT/target).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'evidence':target,'bound_usd':bound,'proposed_charge_usd':charge,'ledger_unchanged':True}))
