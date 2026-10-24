import hashlib,json,os,pathlib,re,subprocess,time,shutil

def train(job_id,config):
 import torch
 root=pathlib.Path('/pg');data=root/'data';audit=json.loads((data/'audit.json').read_text())
 clean=json.loads((data/'clean_audit.json').read_text())
 assert clean['postwrite_exact_document_overlaps']==0 and audit['validation_documents']==50000
 available=clean['retained_training_tokens']
 assert int(config['ITERATIONS'])*524288<=available,'Insufficient distinct training prefix'
 assert torch.cuda.device_count()==8
 names=[torch.cuda.get_device_name(i) for i in range(8)];assert all('H100' in x for x in names),names
 source=pathlib.Path('/work/final_train_gpt.py' if config.get('EXPORT_BEFORE_SCORING')=='1' else '/work/train_gpt.py')
 run=root/'runs'/job_id;run.mkdir(parents=True,exist_ok=False)
 topology=subprocess.check_output(['nvidia-smi','topo','-p2p','n'],text=True)
 smi=subprocess.check_output(['nvidia-smi'],text=True)
 assert topology.count('\tOK')==56,'Expected all 56 directed NVLink peer pairs OK'
 environment={'gpu_names':names,'topology':topology,'nvidia_smi':smi,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'packages':subprocess.check_output(['python','-m','pip','freeze'],text=True),'config':config,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_revision':audit['revision'],'clean_audit_sha256':hashlib.sha256((data/'clean_audit.json').read_bytes()).hexdigest(),'data_derivative':'exact-validation-document matches removed from canonical training only'}
 (run/'environment.json').write_text(json.dumps(environment,indent=2));print(json.dumps({k:v for k,v in environment.items() if k!='packages'}),flush=True)
 # H0: stage identical, verified bytes to ephemeral local storage before torchrun.
 stage_start=time.perf_counter();local=pathlib.Path('/tmp/pg-staged-data');local.mkdir(exist_ok=False)
 needed=int(config['ITERATIONS'])*524288+524288;selected=[];staged_tokens=0
 for f in clean['clean_files']:
  selected.append(f);staged_tokens+=f['tokens']
  if staged_tokens>=needed:break
 assert staged_tokens>=needed
 for f in selected:
  src=data/f['file'];dst=local/src.name;shutil.copyfile(src,dst)
  assert hashlib.sha256(dst.read_bytes()).hexdigest()==f['sha256']
 shutil.copyfile(data/'clean_sp1024/fineweb_val_000000.bin',local/'fineweb_val_000000.bin')
 assert hashlib.sha256((local/'fineweb_val_000000.bin').read_bytes()).hexdigest()==clean['validation_sha256_unchanged']
 staging={'seconds':time.perf_counter()-stage_start,'training_files':len(selected),'training_tokens_available':staged_tokens,'all_hashes_match_clean_audit':True}
 (run/'staging.json').write_text(json.dumps(staging,indent=2));print('local_staging',json.dumps(staging),flush=True)
 env=dict(os.environ,**config,DATA_PATH=str(local),TOKENIZER_PATH=str(data/'datasets/tokenizers/fineweb_1024_bpe.model'),RUN_ID=job_id,NCCL_IB_DISABLE='1',TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1')
 command=['torchrun','--standalone','--nproc_per_node=8',str(source)];start=time.perf_counter();export_observed_elapsed=None
 with (run/'train.log').open('w') as log:
  proc=subprocess.Popen(command,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   time.sleep(10)
   complete_log=(run/'train.log').read_text();tail=complete_log[-1200:];print(tail,flush=True)
   if export_observed_elapsed is None and 'campaign_script_to_export_end_seconds:' in complete_log:
    export_observed_elapsed=time.perf_counter()-start
   if time.perf_counter()-start>700:
    proc.kill();proc.wait();raise TimeoutError('Training subprocess 700s limit; no retry')
 elapsed=time.perf_counter()-start
 text=(run/'train.log').read_text();assert proc.returncode==0,f'training exit {proc.returncode}; tail={text[-3000:]}'
 metrics={}
 for key in ['val_loss','val_bpb']:
  m=re.search(r'final_int8_zlib_roundtrip_exact .*?'+key+r':([0-9.]+)',text);assert m;metrics[key]=float(m[1])
 for key in ['campaign_setup_compile_warmup_seconds','campaign_to_train_end_seconds','campaign_export_seconds','campaign_to_export_end_seconds','campaign_total_seconds']:
  metrics[key]=float(re.findall(key+r':([0-9.]+)',text)[-1])
 if 'campaign_script_to_export_end_seconds:' in text:
  metrics['campaign_script_to_export_end_seconds']=float(re.findall(r'campaign_script_to_export_end_seconds:([0-9.]+)',text)[-1])
 if export_observed_elapsed is not None:metrics['launch_to_export_observed_upper_bound_seconds']=export_observed_elapsed
 metrics['optimization_seconds']=float(re.findall(r'train_time:([0-9]+)ms',text)[-1])/1000
 artifacts=[]
 for name in ['final_model.int8.ptz','train.log','environment.json','staging.json']:
  p=run/name;artifacts.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 model_bytes=(run/'final_model.int8.ptz').stat().st_size
 result={'job_id':job_id,'config':config,'metrics':metrics,'subprocess_wall_seconds':elapsed,'staging':staging,'completed_requested_steps':any(f'step:{config["ITERATIONS"]}/{config["ITERATIONS"]} '+marker in text for marker in ['val_loss:','training_complete']),'model_bytes':model_bytes,'code_bytes':source.stat().st_size,'total_artifact_bytes':model_bytes+source.stat().st_size,'artifact_size_pass':model_bytes+source.stat().st_size<=16000000,'environment':environment,'artifacts':artifacts,'evidence_tier':'fixed-step pilot on decontaminated canonical training with full baseline validation protocol; no official result','baseline_eval_targets':audit['baseline_eval_targets'],'omitted_tail_tokens':audit['baseline_eval_omitted_tail_tokens']}
 (run/'result.json').write_text(json.dumps(result,indent=2));return result
