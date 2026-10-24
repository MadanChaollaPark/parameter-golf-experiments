"""Pinned canonical data download, provenance and complete-document overlap audit."""
import hashlib,json,pathlib,time
REPO='willdepueoai/parameter-golf'
REV='a85b0e6035c3c94bc23685a07c81a8f3bf89db80'
PREFIX='datasets/datasets/fineweb10B_sp1024/'
FILES=[f'{PREFIX}fineweb_val_000000.bin']+[f'{PREFIX}fineweb_train_{i:06d}.bin' for i in range(80)]+['datasets/tokenizers/fineweb_1024_bpe.model','datasets/manifest.json','datasets/docs_selected.source_manifest.json']
def prepare(root='/pg/data'):
 import numpy as np
 from huggingface_hub import HfApi,hf_hub_download
 root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True)
 api=HfApi();metadata=api.get_paths_info(REPO,FILES,repo_type='dataset',revision=REV)
 info={x.path:x for x in metadata};records=[]
 for name in FILES:
  path=pathlib.Path(hf_hub_download(REPO,name,repo_type='dataset',revision=REV,local_dir=root))
  h=hashlib.sha256()
  with path.open('rb') as f:
   while chunk:=f.read(8*1024*1024):h.update(chunk)
  sha=h.hexdigest();m=info[name]
  if m.lfs:assert sha==m.lfs.sha256,(name,'LFS hash mismatch')
  else:
   blob=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0'+path.read_bytes()).hexdigest()
   assert blob==m.blob_id,(name,'Git blob hash mismatch')
  record={'file':name,'bytes':path.stat().st_size,'sha256':sha,'provider_lfs_sha256':m.lfs.sha256 if m.lfs else None,'provider_git_blob':m.blob_id}
  if name.endswith('.bin'):
   header=np.fromfile(path,dtype='<i4',count=256);assert header[0]==20240520 and header[1]==1
   tok=np.memmap(path,dtype='<u2',mode='r',offset=1024);assert len(tok)==header[2]
   assert int(tok.max())<1024
   record.update(tokens=len(tok),bos_count=int(np.count_nonzero(tok==1)),first_token=int(tok[0]),last_token=int(tok[-1]))
  records.append(record);print('verified',name,record.get('tokens',record['bytes']),flush=True)
 val=np.memmap(root/FILES[0],dtype='<u2',mode='r',offset=1024)
 starts=np.flatnonzero(val==1);assert len(starts)==50000 and starts[0]==0
 hashes=set();val_lengths=[]
 ends=list(starts[1:])+[len(val)]
 for a,b in zip(starts,ends):hashes.add(hashlib.sha256(val[a:b]).digest());val_lengths.append(int(b-a))
 overlap=[];complete=0;carry=np.empty(0,dtype='<u2')
 for name in FILES[1:81]:
  tok=np.fromfile(root/name,dtype='<u2',offset=1024)
  tok=np.concatenate((carry,tok)); starts=np.flatnonzero(tok==1);assert starts[0]==0
  for a,b in zip(starts[:-1],starts[1:]):
   complete+=1
   if hashlib.sha256(tok[a:b]).digest() in hashes:overlap.append({'file':name,'start':int(a),'length':int(b-a)})
  carry=tok[starts[-1]:].copy()
 report={'repo':REPO,'revision':REV,'files':records,'validation_documents':50000,'validation_tokens':len(val),
  'validation_unique_document_hashes':len(hashes),'training_complete_documents_checked':complete,'overlapping_complete_documents':overlap,
  'trailing_training_partial_document_tokens':len(carry),'split_policy':'Canonical export flushes validation at document 50000 before starting train; source code reviewed. Downloaded 80 train shards form a prefix; final partial document is not a complete-document hash comparison.',
  'source_docs_cache_downloaded':False,'source_docs_cache_limitation':'48GB source JSONL not redownloaded/re-tokenized; provider-pinned shard hashes, split code/manifest and tokenized-document overlap checks establish bounded provenance, not a new independent raw-corpus rebuild.',
  'baseline_eval_targets':((len(val)-1)//1024)*1024,'baseline_eval_omitted_tail_tokens':(len(val)-1)%1024}
 (root/'audit.json').write_text(json.dumps(report,indent=2))
 assert not overlap,'Canonical downloaded prefix contains full-document train/val overlap; do not train until reviewed'
 return report
