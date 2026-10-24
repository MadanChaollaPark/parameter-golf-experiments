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
 .add_local_file('train_gpt.py','/work/train_gpt.py'))
@app.function(image=image,volumes={'/pg':volume},cpu=(2,2),memory=(8192,8192),timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def prepare():
 import sys
 sys.path.insert(0,'/work')
 from prepare_data import prepare
 result=prepare('/pg/data')
 from correctness_checks import check
 result['correctness']=check('/pg/data');volume.commit();return result

@app.function(image=image,volumes={'/pg':volume},gpu='H100!:8',cpu=(8,8),memory=(32768,32768),timeout=900,startup_timeout=180,retries=0,max_containers=1,scaledown_window=2)
def train(job_id,config):
 import sys
 sys.path.insert(0,'/work')
 from train_remote import train
 try:return train(job_id,config)
 finally:volume.commit()
