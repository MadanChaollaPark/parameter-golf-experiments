"""Extension only. Every invocation must have an extension budget reservation."""
import modal
app=modal.App('parameter-golf-extension-20261024')
volume=modal.Volume.from_name('extension-parameter-golf-20261024',create_if_missing=True)
def install():
 import subprocess,sys
 subprocess.run(['apt-get','update'],check=True,timeout=180)
 subprocess.run(['apt-get','install','-y','--no-install-recommends','gcc','g++','lrzip'],check=True,timeout=300)
 subprocess.run([sys.executable,'-m','pip','install','numpy==2.2.6','sentencepiece==0.2.1','huggingface-hub==0.34.4','brotli==1.1.0','python-minifier==3.1.0'],check=True,timeout=300)
 subprocess.run([sys.executable,'-m','pip','install','--no-deps','https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.03.03-5c98620/flash_attn_3-3.0.0%2B20260303.cu128torch291cxx11abitrue.ceb109-cp39-abi3-linux_x86_64.whl'],check=True,timeout=300)
image=(modal.Image.from_registry('pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime')
 .run_function(install,cpu=(2,2),memory=(8192,8192),timeout=1200,include_source=True)
 .env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_DISABLE_XET':'1','HF_HUB_DISABLE_TELEMETRY':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
 .add_local_dir('extension','/work/extension'))
def claim_once(kind,job_id):
 import datetime,json,pathlib
 p=pathlib.Path('/extension/execution_claims');p.mkdir(exist_ok=True)
 with (p/(kind+'-'+job_id+'.json')).open('x') as f:json.dump({'kind':kind,'job_id':job_id,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_automatic_reexecution':True},f)
 volume.commit()
@app.function(image=image,volumes={'/extension':volume},cpu=(2,2),memory=(16384,16384),timeout=4500,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def prepare(job_id):
 claim_once('prepare',job_id)
 import sys,json
 sys.path.insert(0,'/work/extension')
 from prepare import prepare
 try:return json.loads(json.dumps(prepare()))
 finally:volume.commit()

@app.function(image=image,volumes={'/extension':volume},gpu='H100!:8',cpu=(32,32),memory=(65536,65536),timeout=1710,startup_timeout=90,retries=0,max_containers=1,scaledown_window=2,block_network=True)
def reproduce(job_id):
 claim_once('reproduce',job_id)
 import sys,json
 sys.path.insert(0,'/work/extension')
 from gpu_run import run
 try:return json.loads(json.dumps(run(job_id)))
 finally:volume.commit()

@app.function(image=image,volumes={'/extension':volume},cpu=(2,2),memory=(8192,8192),timeout=60,startup_timeout=60,retries=0,max_containers=1,scaledown_window=2)
def verify(job_id):
 claim_once('verify',job_id)
 import sys,json,pathlib
 sys.path.insert(0,'/work/extension/checks')
 from ngram_order import check
 result=check('/work/extension/upstream-pr2135',pathlib.Path('/extension/pr2135-65doc-counterexample.json'))
 assert result['synthetic_document_count']==65 and result['first_native_queue_document_indices']==[64]
 volume.commit();return result
