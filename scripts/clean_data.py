"""Remove only exact held-out document matches from canonical training prefix.

This is a data-integrity exclusion, not model fitting or validation tuning.
The validation set remains byte-for-byte unchanged. Incomplete final train doc is dropped.
"""
def clean(root='/pg/data'):
 import hashlib,json,pathlib,shutil
 import numpy as np
 root=pathlib.Path(root);src=root/'datasets/datasets/fineweb10B_sp1024';dest=root/'clean_sp1024';dest.mkdir(exist_ok=False)
 val=np.memmap(src/'fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
 vs=np.flatnonzero(val==1);assert len(vs)==50000
 hashes={hashlib.sha256(val[a:b]).digest() for a,b in zip(vs,list(vs[1:])+[len(val)])}
 shutil.copyfile(src/'fineweb_val_000000.bin',dest/'fineweb_val_000000.bin')
 files=[];removed=[];carry=np.empty(0,dtype='<u2');retained_docs=0;kept_tokens=0
 for index,path in enumerate(sorted(src.glob('fineweb_train_*.bin'))):
  tok=np.concatenate((carry,np.fromfile(path,dtype='<u2',offset=1024)));starts=np.flatnonzero(tok==1);assert starts[0]==0
  keep=np.ones(int(starts[-1]),dtype=np.bool_)
  for a,b in zip(starts[:-1],starts[1:]):
   digest=hashlib.sha256(tok[a:b]).digest()
   if digest in hashes:
    keep[a:b]=False;removed.append({'source_file':path.name,'start_in_carry_plus_shard':int(a),'tokens':int(b-a),'document_sha256':digest.hex()})
   else:retained_docs+=1
  carry=tok[starts[-1]:].copy();retained=tok[:len(keep)][keep]
  output=dest/path.name;header=np.zeros(256,dtype='<i4');header[:3]=[20240520,1,len(retained)]
  with output.open('wb') as f:header.tofile(f);retained.tofile(f)
  # Independently re-read materialized output and prove every full document is absent from val.
  check=np.fromfile(output,dtype='<u2',offset=1024);boundaries=np.flatnonzero(check==1)
  for a,b in zip(boundaries,list(boundaries[1:])+[len(check)]):assert hashlib.sha256(check[a:b]).digest() not in hashes
  sha=hashlib.sha256(output.read_bytes()).hexdigest();kept_tokens+=len(retained)
  files.append({'file':str(output.relative_to(root)),'bytes':output.stat().st_size,'tokens':len(retained),'sha256':sha})
  print('cleaned',index,'removed_docs_so_far',len(removed),'retained_tokens',kept_tokens,flush=True)
 original=json.loads((root/'audit.json').read_text());assert len(removed)==len(original['overlapping_complete_documents'])
 original_val=next(x['sha256'] for x in original['files'] if 'fineweb_val_' in x['file'])
 assert hashlib.sha256((dest/'fineweb_val_000000.bin').read_bytes()).hexdigest()==original_val
 report={'source_revision':original['revision'],'validation_sha256_unchanged':original_val,'validation_documents':50000,'training_data_path':'clean_sp1024','removed_documents':removed,'removed_document_count':len(removed),'removed_tokens':sum(x['tokens'] for x in removed),'dropped_incomplete_final_document_tokens':len(carry),'retained_complete_documents':retained_docs,'retained_training_tokens':kept_tokens,'clean_files':files,'postwrite_exact_document_overlaps':0,'data_change_reason':'Exclude exact held-out document token hashes; validation remains untouched. Not an exact official baseline reproduction.'}
 (root/'clean_audit.json').write_text(json.dumps(report,indent=2));return report
