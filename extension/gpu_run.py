"""One bounded allocation: full native training/export, then fresh artifact TTT."""
import datetime,hashlib,json,os,pathlib,re,shutil,signal,socket,subprocess,sys,time
ROOT=pathlib.Path('/extension')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def run(job_id):
 import torch
 begin=time.monotonic();deadline=begin+1650
 assert torch.cuda.device_count()==8 and all('H100' in torch.cuda.get_device_name(i) for i in range(8))
 topology=subprocess.check_output(['nvidia-smi','topo','-p2p','n'],text=True);assert topology.count('\tOK')==56
 out=ROOT/'runs'/job_id;out.mkdir(parents=True,exist_ok=False)
 prep=json.loads((ROOT/'prepare-result.json').read_text());assert prep['byte_audit']['sidecar_mismatch_count']==0 and prep['native_chunk_ownership_targets']==47853343
 assert prep['ngram_counterexample']['cross_document_dependency_demonstrated']
 source=pathlib.Path('/work/extension/upstream-pr2014/train_gpt.py');assert sha(source)=='a496bb37e25bc7ccaa30c4d89253376b4bc8929ffcd1d3f464367acf13fcebc9'
 config=json.loads(pathlib.Path('/work/extension/configs/published-pr2014-seed42.json').read_text())
 env=dict(os.environ,**config,NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',NCCL_IB_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',HF_HUB_OFFLINE='1',EXTENSION_OUTPUT=str(out),EXTENSION_NATIVE_SOURCE=str(source),EXTENSION_DATA_AUDIT=str(ROOT/'prepare-result.json'))
 staged=pathlib.Path('/tmp/extension-native-data');staged.mkdir()
 src=pathlib.Path(prep['selected_train_path']);staging_start=time.perf_counter();stage_files=[]
 records={f['file']:f for f in prep['clean_files']} if prep['removed_count'] else {pathlib.Path(f['file']).name:f for f in json.loads((ROOT/'download-manifest.json').read_text())}
 # Stage the expected consumed prefix; all80 original file names remain available for native random training-only GPTQ calibration.
 for i,path in enumerate(sorted(src.glob('fineweb_train_*.bin'))):
  dest=staged/path.name
  if i<48:
   shutil.copyfile(path,dest);digest=sha(dest);assert digest==records[path.name]['sha256']
  else:dest.symlink_to(path);digest=records[path.name]['sha256']
  stage_files.append({'file':path.name,'sha256':digest,'ephemeral_copy':i<48})
 for name in ['fineweb_val_000000.bin','fineweb_val_bytes_000000.bin']:shutil.copyfile(src/name,staged/name)
 tokenizer=ROOT/'data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
 shutil.copyfile(tokenizer,out/tokenizer.name)
 env.update(DATA_PATH=str(staged),DATA_DIR=str(staged),TOKENIZER_PATH=str(out/tokenizer.name),ARTIFACT_DIR=str(out),RUN_ID=job_id,TTT_ENABLED='0',EXTENSION_PHASE='training')
 metadata={'job_id':job_id,'source_sha256':sha(source),'published_config':config,'operational_override':'TTT_ENABLED=0 for native train/export process; fresh TTT_EVAL_ONLY=1 process runs unchanged native evaluation with TTT_ENABLED=1','gpu_names':[torch.cuda.get_device_name(i) for i in range(8)],'topology':topology,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'staging_seconds':time.perf_counter()-staging_start,'staging_files':stage_files,'data_audit_sha256':sha(ROOT/'prepare-result.json'),'data_removed_overlap_documents':prep['removed_count'],'torch':str(torch.__version__),'cuda':torch.version.cuda,'processes':[]}
 (out/'environment.json').write_text(json.dumps(metadata,indent=2))
 def phase(name):
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  command=['torchrun','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1',f'--master_port={port}','--max_restarts=0','/work/extension/native_driver.py']
  started=time.monotonic();log=out/(name+'.log')
  with log.open('w') as stream:
   process=subprocess.Popen(command,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   while process.poll() is None:
    time.sleep(5)
    print(log.read_text(errors='replace')[-1400:],flush=True)
    if time.monotonic()>deadline:
     os.killpg(process.pid,signal.SIGKILL);process.wait();raise TimeoutError('Combined allocation child deadline; no retry')
  meta={'phase':name,'returncode':process.returncode,'subprocess_wall_seconds':time.monotonic()-started,'command':command};metadata['processes'].append(meta);(out/'environment.json').write_text(json.dumps(metadata,indent=2))
  assert process.returncode==0,'Native phase failed; preserve logs and persisted artifacts'
 phase('training')
 artifact=out/'final_model.int6.ptz';assert artifact.exists();model_sha=sha(artifact)
 # Preserve the checkpoint before evaluation; remove all physical training shards
 # from this allocation and volume so eval cannot access them even accidentally.
 import modal
 modal.Volume.from_name('extension-parameter-golf-20261024').commit()
 evaluation=ROOT/'evaluation';evaluation.mkdir()
 for name in ['fineweb_val_000000.bin','fineweb_val_bytes_000000.bin']:shutil.copyfile(staged/name,evaluation/name)
 shutil.rmtree(staged);shutil.rmtree(ROOT/'data');shutil.rmtree(ROOT/'clean')
 assert not list(ROOT.rglob('fineweb_train_*.bin'))
 env.update(DATA_PATH=str(evaluation),DATA_DIR=str(evaluation),TTT_ENABLED='1',TTT_EVAL_ONLY='1',EXTENSION_PHASE='evaluation')
 phase('evaluation')
 assert sha(artifact)==model_sha
 logs=(out/'training.log').read_text()+(out/'evaluation.log').read_text()
 scores=re.findall(r'quantized_ttt_phased val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:([0-9.]+)ms',logs);assert len(scores)==1
 events=[json.loads(x) for x in (out/'evaluation-events.jsonl').read_text().splitlines()]
 totals=next(x for x in events if x['event']=='ttt_scored_totals')
 result={'job_id':job_id,'native_pr':2014,'native_head':'fcd0b8357f58996a931a515c7da3ee0c999abc1e','val_loss':float(scores[0][0]),'val_bpb':float(scores[0][1]),'native_ttt_eval_seconds':float(scores[0][2])/1000,'target_tokens':totals['target_tokens'],'original_bytes':totals['original_bytes'],'model_bytes':artifact.stat().st_size,'model_sha256':model_sha,'metadata':metadata,'allocation_function_seconds':time.monotonic()-begin,'fresh_artifact_process':True,'all_training_data_removed_before_evaluation':True,'compressed_artifact_unchanged_after_ttt':True,'no_official_result':True}
 (out/'result.json').write_text(json.dumps(result,indent=2));return result
