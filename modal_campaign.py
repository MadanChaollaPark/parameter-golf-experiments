"""Bounded ordinary Modal Functions. Use scripts/run_job.py; never bypass reservations."""
import modal
app=modal.App('parameter-golf-controlled-20261024')
volume=modal.Volume.from_name('parameter-golf-controlled-20261024',create_if_missing=True)
image=(modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime')
 .run_commands('apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*',cpu=2,memory=8192,timeout=600)
 .run_commands('python -m pip install numpy==2.2.6 sentencepiece==0.2.1 huggingface-hub==0.34.4',cpu=2,memory=8192,timeout=600)
 .env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_XET':'1','PYTHONUNBUFFERED':'1'})
 .add_local_dir('vendor','/work/vendor')
 .add_local_file('scripts/prepare_data.py','/work/prepare_data.py'))
@app.function(image=image,volumes={'/pg':volume},cpu=(2,2),memory=(8192,8192),timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def prepare():
 import sys
 sys.path.insert(0,'/work')
 from prepare_data import prepare
 result=prepare('/pg/data');volume.commit();return result
