import hashlib,json,os,pathlib,shutil,subprocess,time

def stage(job_id):
 source=pathlib.Path('/pg/runs')/job_id;out=pathlib.Path('/evaluation')/job_id;out.mkdir(parents=True,exist_ok=False)
 result=json.loads((source/'result.json').read_text());audit=json.loads(pathlib.Path('/pg/data/audit.json').read_text())
 files={'final_model.int8.ptz':source/'final_model.int8.ptz','fineweb_val_000000.bin':pathlib.Path('/pg/data/datasets/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'),'fineweb_1024_bpe.model':pathlib.Path('/pg/data/datasets/tokenizers/fineweb_1024_bpe.model')}
 hashes={}
 for name,src in files.items():
  shutil.copyfile(src,out/name);hashes[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
 assert hashes['final_model.int8.ptz']==result['artifacts'][0]['sha256']
 for name in ['fineweb_val_000000.bin','fineweb_1024_bpe.model']:
  expected=next(f['sha256'] for f in audit['files'] if f['file'].endswith('/'+name));assert hashes[name]==expected
 # Independently decode every validation document for the original-byte denominator.
 import numpy as np
 import sentencepiece as spm
 sp=spm.SentencePieceProcessor(model_file=str(out/'fineweb_1024_bpe.model'))
 tokens=np.memmap(out/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
 starts=np.flatnonzero(tokens==sp.bos_id());assert len(starts)==50000 and starts[0]==0
 decoded_bytes=0
 for i,a in enumerate(starts):
  b=starts[i+1] if i+1<len(starts) else len(tokens)
  decoded_bytes+=len(sp.decode(tokens[a:b].tolist()).encode('utf-8'))
 (out/'config.json').write_text(json.dumps(result['config'],indent=2))
 manifest={'job_id':job_id,'model_sha256':hashes['final_model.int8.ptz'],'expected_baseline_bpb':result['metrics']['val_bpb'],'file_sha256':hashes,'data_revision':audit['revision'],'full_validation_decoded_bytes':decoded_bytes,'validation_documents_decoded':len(starts)}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 return {'job_id':job_id,'manifest':manifest,'staged_names':sorted(p.name for p in out.iterdir()),'no_training_shards_staged':not any(out.glob('*train*'))}

def evaluate(job_id):
 import torch
 assert torch.cuda.device_count()==8 and all('H100' in torch.cuda.get_device_name(i) for i in range(8))
 root=pathlib.Path('/evaluation')/job_id
 assert not pathlib.Path('/pg').exists(),'Training volume must not be mounted'
 assert not list(root.glob('*train*'))
 environment={'gpu_names':[torch.cuda.get_device_name(i) for i in range(8)],'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'topology':subprocess.check_output(['nvidia-smi','topo','-m'],text=True)}
 env=dict(os.environ,NCCL_IB_DISABLE='1',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',HF_HUB_OFFLINE='1')
 command=['torchrun','--standalone','--nproc_per_node=8','/work/evaluate.py','--directory',str(root)]
 start=time.perf_counter()
 with (root/'eval.log').open('w') as f:
  p=subprocess.Popen(command,cwd='/work',env=env,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   time.sleep(5);print((root/'eval.log').read_text()[-1500:],flush=True)
   if time.perf_counter()-start>370:p.kill();p.wait();raise TimeoutError('Offline evaluation 370s limit')
 assert p.returncode==0,(root/'eval.log').read_text()[-4000:]
 d=json.loads((root/'evaluation.json').read_text());d['subprocess_wall_seconds']=time.perf_counter()-start;d['environment']=environment
 d['training_script_bytes']=pathlib.Path('/work/train_gpt.py').stat().st_size
 d['evaluation_code_bytes']=sum(pathlib.Path('/work',f).stat().st_size for f in ['evaluate.py','eval_windows.py'])
 d['model_bytes']=(root/'final_model.int8.ptz').stat().st_size
 d['configuration_bytes']=sum((root/f).stat().st_size for f in ['manifest.json','config.json'])
 d['artifact_bytes']=d['model_bytes']+d['training_script_bytes']+d['evaluation_code_bytes']+d['configuration_bytes']
 d['artifact_bytes_including_canonical_tokenizer']=d['artifact_bytes']+(root/'fineweb_1024_bpe.model').stat().st_size
 d['under_600_seconds']=d['subprocess_wall_seconds']<600
 d['artifact_size_pass']=d['artifact_bytes']<=16000000
 (root/'evaluation.json').write_text(json.dumps(d,indent=2));return d
