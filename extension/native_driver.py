"""Observation-only wrapper around unmodified PR2014 native functions.

Called in separate torchrun processes for native training/export and native
TTT_EVAL_ONLY reconstruction. No model, optimizer, quantizer or scorer replacement.
"""
import datetime,hashlib,importlib.util,inspect,json,os,pathlib,sys,time
START=time.perf_counter();ROOT=pathlib.Path(os.environ['EXTENSION_OUTPUT']);ROOT.mkdir(exist_ok=True)
SOURCE=pathlib.Path(os.environ['EXTENSION_NATIVE_SOURCE']);PHASE=os.environ['EXTENSION_PHASE']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='a496bb37e25bc7ccaa30c4d89253376b4bc8929ffcd1d3f464367acf13fcebc9'
sys.path.insert(0,str(SOURCE.parent));spec=importlib.util.spec_from_file_location('native_leading',SOURCE);native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
import torch
rank=int(os.environ.get('RANK','0'));events=[]
def record(event,**values):
 d={'event':event,'phase':PHASE,'rank':rank,'script_elapsed_seconds':time.perf_counter()-START,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**values}
 if rank==0:
  with (ROOT/(PHASE+'-events.jsonl')).open('a') as f:f.write(json.dumps(d)+'\n')
  print('EXTENSION_EVENT',json.dumps(d),flush=True)
def wrap(name):
 original=getattr(native,name)
 def measured(*args,**kwargs):
  torch.cuda.synchronize();t=time.perf_counter();record(name+'_begin')
  result=original(*args,**kwargs)
  torch.cuda.synchronize();record(name+'_end',wall_seconds=time.perf_counter()-t)
  return result
 setattr(native,name,measured)
for name in ['train_model','serialize','deserialize','eval_val_ttt_phased','collect_hessians','_pack_pergroup']:wrap(name)
original_sums=native._loss_bpb_from_sums
def sums(loss_sum,token_count,byte_sum):
 record('ttt_scored_totals',nll_sum=float(loss_sum),target_tokens=int(token_count),original_bytes=int(byte_sum))
 assert int(token_count)==47853343
 expected=json.loads(pathlib.Path(os.environ['EXTENSION_DATA_AUDIT']).read_text())['byte_audit']['sidecar_bytes']
 assert int(byte_sum)==expected
 return original_sums(loss_sum,token_count,byte_sum)
native._loss_bpb_from_sums=sums
original_global=native.train_val_ttt_global_sgd_distributed
def global_ttt(*args,**kwargs):
 frame=inspect.currentframe().f_back;selected=frame.f_locals['scored_docs_for_global'];gathered=frame.f_locals['gathered_scored_docs'];seen={tuple(d) for docs in gathered if docs for d in docs}
 assert len(selected)<=2500 and all(tuple(d) in seen for d in selected)
 record('global_ttt_scored_document_gate',selected_documents=len(selected),all_selected_already_scored=True,document_indices_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest())
 return original_global(*args,**kwargs)
native.train_val_ttt_global_sgd_distributed=global_ttt
# Fail closed if the separate evaluation process tries to open training data.
if PHASE=='evaluation':
 assert os.environ['TTT_EVAL_ONLY']=='1'
 assert not list(pathlib.Path(os.environ['DATA_PATH']).glob('fineweb_train_*.bin'))
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(args[0]);assert 'fineweb_train_' not in p,'Evaluation attempted training-data access'
 sys.addaudithook(audit)
 record('evaluation_isolation',fresh_process=True,train_files_in_data_path=0,training_file_open_guard=True)
record('native_main_begin',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),torch=str(torch.__version__),cuda=torch.version.cuda)
native.main()
record('native_main_end')
