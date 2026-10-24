"""Bounded ordinary Modal Functions. Use scripts/run_job.py; never bypass reservations."""
import modal
app=modal.App('parameter-golf-controlled-20261024')
volume=modal.Volume.from_name('parameter-golf-controlled-20261024',create_if_missing=True)
def install_packages():
 import subprocess
 subprocess.run(['apt-get','update'],check=True,timeout=180)
 subprocess.run(['apt-get','install','-y','--no-install-recommends','gcc','g++'],check=True,timeout=300)
 subprocess.run(['python','-m','pip','install','numpy==2.2.6','sentencepiece==0.2.1','huggingface-hub==0.34.4'],check=True,timeout=300)
image=(modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime')
 .run_function(install_packages,cpu=(2,2),memory=(8192,8192),timeout=900,include_source=True)
 .env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_XET':'1','PYTHONUNBUFFERED':'1'})
 .add_local_dir('vendor','/work/vendor')
 .add_local_file('scripts/prepare_data.py','/work/prepare_data.py')
 .add_local_file('scripts/correctness_checks.py','/work/correctness_checks.py')
 .add_local_file('scripts/train_remote.py','/work/train_remote.py')
 .add_local_file('submission/train_gpt.py','/work/train_gpt.py')
 .add_local_file('evaluate.py','/work/evaluate.py')
 .add_local_file('eval_windows.py','/work/eval_windows.py')
 .add_local_file('scripts/evaluation_remote.py','/work/evaluation_remote.py')
 .add_local_file('scripts/clean_data.py','/work/clean_data.py'))
@app.function(image=image,volumes={'/pg':volume},cpu=(2,2),memory=(8192,8192),timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def prepare():
 import sys
 sys.path.insert(0,'/work')
 from prepare_data import prepare
 result=prepare('/pg/data')
 from correctness_checks import check
 result['correctness']=check('/pg/data');volume.commit();return result

@app.function(image=image,volumes={'/pg':volume},gpu='H100!:8',cpu=(32,32),memory=(65536,65536),timeout=800,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2)
def train(job_id,config):
 import sys
 sys.path.insert(0,'/work')
 from train_remote import train
 try:
  import json
  return json.loads(json.dumps(train(job_id,config)))
 finally:volume.commit()

evaluation_volume=modal.Volume.from_name('parameter-golf-evaluation-20261024',create_if_missing=True)
@app.function(image=image,volumes={'/pg':volume,'/evaluation':evaluation_volume},cpu=(2,2),memory=(8192,8192),timeout=240,startup_timeout=60,retries=0,max_containers=1,scaledown_window=2)
def stage(job_id):
 import sys
 sys.path.insert(0,'/work')
 from evaluation_remote import stage
 result=stage(job_id);evaluation_volume.commit();return result
@app.function(image=image,volumes={'/evaluation':evaluation_volume},gpu='H100!:8',cpu=(32,32),memory=(65536,65536),timeout=400,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,block_network=True)
def evaluate(job_id):
 import sys
 sys.path.insert(0,'/work')
 from evaluation_remote import evaluate
 try:
  import json
  return json.loads(json.dumps(evaluate(job_id)))
 finally:evaluation_volume.commit()

@app.function(image=image,volumes={'/pg':volume},cpu=(2,2),memory=(8192,8192),timeout=1200,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2)
def clean():
 import sys
 sys.path.insert(0,'/work')
 from clean_data import clean
 from correctness_checks import check
 try:
  result=clean('/pg/data');result['correctness']=check('/pg/data');return result
 finally:volume.commit()

@app.function(image=image,gpu='H100!:8',cpu=(2,2),memory=(8192,8192),timeout=60,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2)
def hardware():
 import subprocess,time,torch
 start=time.perf_counter();records=[]
 for command in [['nvidia-smi'],['nvidia-smi','--query-gpu=name,memory.total,power.max_limit,driver_version','--format=csv'],['nvidia-smi','topo','-m'],['nvidia-smi','topo','-p2p','n'],['nvidia-smi','nvlink','--status']]:
  r=subprocess.run(command,capture_output=True,text=True,timeout=8);records.append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
 peer=[[bool(torch.cuda.can_device_access_peer(i,j)) if i!=j else True for j in range(len(names))] for i in range(len(names))]
 checks=[]
 for i in range(len(names)):
  with torch.cuda.device(i):
   x=torch.ones((16,16),device='cuda');y=x@x;torch.cuda.synchronize();checks.append(float(y[0,0])==16.)
 return {'gpu_names':names,'cuda_version':torch.version.cuda,'peer_access':peer,'small_cuda_matmul_checks':checks,'commands':records,'elapsed_seconds':time.perf_counter()-start,'purpose':'Read-only hardware characterization after topology CLI failure; no training retry'}
