"""Explicit reproduction on an existing CUDA machine; no cloud provisioning."""
import argparse,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
c=json.loads(pathlib.Path(a.config).read_text());data=pathlib.Path(a.data).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,**c['config'],RUN_ID=c['job_id'],DATA_PATH=str(data/'clean_sp1024'),TOKENIZER_PATH=str(data/'datasets/tokenizers/fineweb_1024_bpe.model'),NCCL_IB_DISABLE='1',TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
subprocess.run(['torchrun','--standalone','--nproc_per_node=8',str(ROOT/'submission/train_gpt.py')],cwd=out,env=env,check=True)
