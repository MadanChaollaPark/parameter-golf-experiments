"""Preserve final evaluation evidence before deleting campaign volumes."""
import hashlib,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['MODAL_CONFIG_PATH']='/home/azureuser/parameter-golf-control/private/modal.toml'
import modal
v=modal.Volume.from_name('parameter-golf-evaluation-20261024')
for job in sys.argv[1:]:
 destination=ROOT/'artifacts'/job;destination.mkdir(parents=True,exist_ok=True);records=[]
 for name in ['config.json','manifest.json','evaluation.json','partial_evaluation.json','reference_fidelity.json','eval.log']:
  data=b''.join(v.read_file(job+'/'+name));(destination/name).write_bytes(data)
  records.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 manifest=json.loads((destination/'manifest.json').read_text());input_hashes={}
 for name in ['fineweb_val_000000.bin','fineweb_1024_bpe.model']:
  digest=hashlib.sha256()
  for data in v.read_file(job+'/'+name):digest.update(data)
  input_hashes[name]=digest.hexdigest();assert input_hashes[name]==manifest['file_sha256'][name]
 (destination/'evaluation_input_integrity_after_run.json').write_text(json.dumps({'verified_after_evaluation':True,'sha256':input_hashes,'raw_input_bytes_not_stored_locally':True},indent=2)+'\n')
 result=json.loads((destination/'evaluation.json').read_text())
 assert hashlib.sha256((destination/'final_model.int8.ptz').read_bytes()).hexdigest()==result['model_sha256']
 assert result['frozen_state_hash_before']==result['frozen_state_hash_after']
 (destination/'evaluation_download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 print('Verified evaluation and checkpoint linkage:',job)
