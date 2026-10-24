"""Cheap CPU checks; use only training/synthetic tokens for model checks."""
def check(root='/pg/data'):
 import importlib.util,json,pathlib,hashlib,math
 import numpy as np
 import sentencepiece as spm
 import torch
 torch.set_num_threads(1)
 spec=importlib.util.spec_from_file_location('baseline','/work/vendor/train_gpt.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 torch.manual_seed(20261024)
 model=m.GPT(vocab_size=1024,num_layers=2,model_dim=32,num_heads=4,num_kv_heads=2,mlp_mult=2,tie_embeddings=True,tied_embed_init_std=.005,logit_softcap=30.,rope_base=10000.,qk_gain_init=1.5)
 # Activate projections so causality cannot pass merely because residual updates are zero.
 with torch.no_grad():
  for block in model.blocks:
   block.attn.proj.weight.normal_(std=.02);block.mlp.proj.weight.normal_(std=.02)
 model.eval();x=torch.randint(0,1024,(2,32));y=torch.randint(0,1024,(2,32));y[:,12:]=-100
 with torch.no_grad():
  l1=model(x,y);x2=x.clone();x2[:,12:]=torch.randint(0,1024,(2,20));l2=model(x2,y)
 assert torch.allclose(l1,l2,atol=1e-6,rtol=0),'Future tokens influenced earlier predictions'
 loss=model(x,torch.randint(0,1024,(2,32)));loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 state={'test.weight':torch.randn(512,512),'test.scale':torch.ones(12)};q,stats=m.quantize_state_dict_int8(state);deq=m.dequantize_state_dict_int8(q)
 assert deq.keys()==state.keys() and all(torch.isfinite(x).all() for x in deq.values())
 mse=float((state['test.weight']-deq['test.weight']).square().mean());assert mse<.001
 root=pathlib.Path(root);sp=spm.SentencePieceProcessor(model_file=str(root/'datasets/tokenizers/fineweb_1024_bpe.model'))
 base,leading,boundary=m.build_sentencepiece_luts(sp,1024,torch.device('cpu'))
 val=np.memmap(root/'datasets/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',dtype='<u2',mode='r',offset=1024)
 starts=np.flatnonzero(val==1);indices=np.random.default_rng(20261024).choice(len(starts)-1,size=100,replace=False);checks=[]
 for i in indices:
  ids=torch.tensor(val[starts[i]:starts[i+1]].astype('int64'));b=base[ids[1:]]+(leading[ids[1:]] & ~boundary[ids[:-1]]).to(torch.int16)
  measured=int(b.sum());decoded=len(sp.decode(ids.tolist()).encode('utf-8'));assert measured==decoded,(int(i),measured,decoded)
  checks.append({'doc_index':int(i),'lut_bytes':measured,'decoded_bytes':decoded})
 return {'causality_loss_abs_diff':float(abs(l1-l2)),'finite_gradients':True,'quantization_synthetic_mse':mse,'quantization_uses_validation':False,'token_byte_decode_checks':checks,'validation_used_only_for_metric_correctness':True}
