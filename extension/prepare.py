"""CPU-only dependencies, pinned CaseOps data, and full document/byte audit."""
import hashlib,json,pathlib,sys,time,subprocess
REPO='romeerp/parameter-golf-caseops-v1';REV='889572bcefaa6b80257fd266d6e99ed9624ea9fa'
PREFIX='datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/'
TOKENIZER='datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def prepare():
 import numpy as np,sentencepiece as spm,torch
 from huggingface_hub import HfApi,hf_hub_download
 root=pathlib.Path('/extension');root.mkdir(exist_ok=True);data=root/'data';data.mkdir(exist_ok=True)
 sys.path.insert(0,'/work/extension/checks');from ngram_order import check
 ngram=check('/work/extension/upstream-pr2135',root/'pr2135-ngram-order-counterexample.json');print(json.dumps(ngram),flush=True)
 # Import the actual selected implementation and binary dependency before any GPU allocation.
 sys.path.insert(0,'/work/extension/upstream-pr2014')
 import flash_attn_interface
 import train_gpt as native
 torch.set_num_threads(1)
 from lossless_caps import decode_lossless_caps_v2,encode_lossless_caps_v2
 files=[PREFIX+'fineweb_val_000000.bin',PREFIX+'fineweb_val_bytes_000000.bin',TOKENIZER,'datasets/manifest.json','datasets/docs_selected.source_manifest.json','datasets/tokenizer_config.export.json']+[PREFIX+f'fineweb_train_{i:06d}.bin' for i in range(80)]
 info={x.path:x for x in HfApi().get_paths_info(REPO,files,repo_type='dataset',revision=REV)};records=[]
 for name in files:
  p=pathlib.Path(hf_hub_download(REPO,name,repo_type='dataset',revision=REV,local_dir=data));digest=sha(p);m=info[name]
  if m.lfs:assert digest==m.lfs.sha256
  else:assert hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==m.blob_id
  records.append({'file':name,'bytes':p.stat().st_size,'sha256':digest,'provider_lfs_sha256':m.lfs.sha256 if m.lfs else None,'provider_git_blob':m.blob_id})
  print('verified',name,flush=True)
 (root/'download-manifest.json').write_text(json.dumps(records,indent=2))
 assert sha(data/TOKENIZER)==sha('/work/extension/upstream-pr2014/tokenizers/'+pathlib.Path(TOKENIZER).name)
 def read(p):
  header=np.fromfile(p,dtype='<i4',count=256);assert header[0]==20240520 and header[1]==1
  t=np.fromfile(p,dtype='<u2',offset=1024);assert len(t)==header[2];return t
 val=read(data/files[0]);side=read(data/files[1]);assert len(val)==47853344 and len(side)==len(val)
 starts=np.flatnonzero(val==1);assert len(starts)==50000 and starts[0]==0
 sp=spm.SentencePieceProcessor(model_file=str(data/TOKENIZER));assert sp.vocab_size()==8192
 hashes=set();decoded_bytes=0;sidecar_mismatches=[];roundtrip_mismatches=[]
 for index,(a,b) in enumerate(zip(starts,list(starts[1:])+[len(val)])):
  hashes.add(hashlib.sha256(val[a:b]).digest())
  transformed=sp.decode(val[a+1:b].tolist());original=decode_lossless_caps_v2(transformed)
  size=len(original.encode('utf-8'));expected=int(side[a:b].sum());decoded_bytes+=size
  if size!=expected:sidecar_mismatches.append({'doc':index,'decoded':size,'sidecar':expected})
  if encode_lossless_caps_v2(original)!=transformed:roundtrip_mismatches.append(index)
  if index%10000==0:print('decoded validation doc',index,flush=True)
 byte_audit={'validation_docs':50000,'validation_tokens':len(val),'prediction_targets':len(val)-1,'sidecar_bytes':int(side[1:].sum()),'independently_decoded_bytes':decoded_bytes,'sidecar_mismatches':sidecar_mismatches[:20],'sidecar_mismatch_count':len(sidecar_mismatches),'caseops_roundtrip_mismatch_count':len(roundtrip_mismatches)}
 (root/'byte-audit.json').write_text(json.dumps(byte_audit,indent=2));assert not sidecar_mismatches,'Original byte decode/sidecar mismatch'
 # Audit full native document/chunk ownership by calling upstream pure helpers.
 native.BOS_ID=1;docs=native._find_docs(torch.from_numpy(val.astype('int64')));assert len(docs)==50000
 owned=0
 for start,length in docs:
  pred=length-1;cs=8 if length<=256 else (24 if length<=2000 else 48);nc=(pred+cs-1)//cs;previous=0
  for ci in range(nc):
   ws,wl,offset,cl=native._compute_chunk_window(ci,pred,nc,cs,3072)
   assert ws+offset==previous and wl==offset+cl and cl>0 and wl<=3072
   previous+=cl
  assert previous==pred;owned+=pred
 assert owned==len(val)-1
 # Full train stream has a manifest-recorded complete final document; count proves scope.
 clean=root/'clean';clean.mkdir(exist_ok=False);carry=np.empty(0,dtype='<u2');removed=[];complete=0;retained=0;outputs=[]
 for i in range(80):
  name=PREFIX+f'fineweb_train_{i:06d}.bin';tok=np.concatenate((carry,read(data/name)));assert int(tok.max())<8192
  bounds=np.flatnonzero(tok==1);assert bounds[0]==0
  ends=list(bounds[1:])+([len(tok)] if i==79 else [])
  limit=len(tok) if i==79 else int(bounds[-1]);keep=np.ones(limit,dtype=bool)
  for a,b in zip(bounds,ends):
   complete+=1;digest=hashlib.sha256(tok[a:b]).digest()
   if digest in hashes:keep[a:b]=False;removed.append({'source_file':name,'start':int(a),'tokens':int(b-a),'sha256':digest.hex()})
   else:retained+=1
  carry=tok[limit:].copy();kept=tok[:limit][keep];out=clean/pathlib.Path(name).name
  header=np.zeros(256,dtype='<i4');header[:3]=[20240520,1,len(kept)]
  with out.open('wb') as f:header.tofile(f);kept.tofile(f)
  reread=read(out);bs=np.flatnonzero(reread==1)
  for a,b in zip(bs,list(bs[1:])+[len(reread)]):assert hashlib.sha256(reread[a:b]).digest() not in hashes
  outputs.append({'file':out.name,'bytes':out.stat().st_size,'tokens':len(kept),'sha256':sha(out)})
  print('audited training shard',i,'overlap_docs',len(removed),flush=True)
 assert complete==8181945 and len(carry)==0
 import shutil
 for name in ['fineweb_val_000000.bin','fineweb_val_bytes_000000.bin']:shutil.copyfile(data/PREFIX/name,clean/name)
 report={'repo':REPO,'revision':REV,'provider_manifest_split_docs':50000,'full_train_documents':complete,'validation_unique_document_hashes':len(hashes),'removed_exact_overlap_documents':removed,'removed_count':len(removed),'removed_tokens':sum(x['tokens'] for x in removed),'retained_documents':retained,'clean_files':outputs,'postwrite_exact_overlap_count':0,'selected_train_path':str(clean) if removed else str(data/PREFIX),'provenance_limit':'Provider pinned tokenized export, manifest/source split and complete tokenized-document audit; no raw48GB JSONL re-export, no near-duplicate exclusion, no independently reconstructed historical tokenizer fit.','byte_audit':byte_audit,'native_chunk_ownership_targets':owned,'ngram_counterexample':ngram,'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'lrzip':subprocess.run(['lrzip','--version'],capture_output=True,text=True).stdout,'flash_attn_import':True}}
 (root/'prepare-result.json').write_text(json.dumps(report,indent=2));return report
