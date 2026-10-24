"""Fresh offline evaluation of a compressed baseline artifact. torchrun --nproc_per_node=8."""
import time
SCRIPT_START=time.perf_counter()
import argparse,hashlib,io,json,math,os,pathlib,zlib
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import train_gpt as m
from eval_windows import windows,prove_coverage

def state_hash(model):
 h=hashlib.sha256()
 for name,t in model.state_dict().items():
  h.update(name.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def score(model,tokens,n_targets,stride,rank,world,device,base,leading,boundary):
 spec=list(windows(n_targets,1024,stride));local=spec[len(spec)*rank//world:len(spec)*(rank+1)//world]
 totals=torch.zeros(3,dtype=torch.float64,device=device);batch_size=64
 t0=time.perf_counter();model.eval()
 with torch.inference_mode():
  for offset in range(0,len(local),batch_size):
   batch=local[offset:offset+batch_size]
   x=torch.zeros((batch_size,1024),dtype=torch.int64)
   y=torch.full_like(x,-100);count=0;byte_count=0
   for row,(start,a,b) in enumerate(batch):
    length=b-start;x[row,:length]=tokens[start:b]
    y[row,a-start:length]=tokens[a+1:b+1];count+=b-a
    prev=tokens[a:b];target=tokens[a+1:b+1]
    byte_count+=int((base[target]+(leading[target]&~boundary[prev]).to(torch.int16)).sum())
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x.to(device),y.to(device))
   totals[0]+=loss.double()*count;totals[1]+=count;totals[2]+=byte_count
   if rank==0 and offset%(batch_size*250)==0:print(f'eval stride={stride} windows={offset}/{len(local)}',flush=True)
 dist.all_reduce(totals);torch.cuda.synchronize()
 loss_sum,count,byte_count=totals.tolist();assert int(count)==n_targets
 return {'stride':stride,'context':1024,'nll_sum':loss_sum,'target_tokens':int(count),'original_bytes':int(byte_count),'val_loss':loss_sum/count,'val_bpb':loss_sum/math.log(2)/byte_count,'scoring_seconds_including_first_compile':time.perf_counter()-t0,'coverage':prove_coverage(n_targets,1024,stride)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=pathlib.Path(a.directory)
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
 assert world==8 and all('H100' in torch.cuda.get_device_name(i) for i in range(8))
 dist.init_process_group('nccl',device_id=device)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
 enable_cudnn_sdp(False);enable_flash_sdp(True);enable_math_sdp(False);enable_mem_efficient_sdp(False)
 config=json.loads((root/'config.json').read_text())
 args=m.Hyperparameters()
 model=m.GPT(args.vocab_size,args.num_layers,args.model_dim,args.num_heads,args.num_kv_heads,args.mlp_mult,args.tie_embeddings,args.tied_embed_init_std,args.logit_softcap,args.rope_base,float(config['QK_GAIN_INIT'])).to(device).bfloat16()
 for module in model.modules():
  if isinstance(module,m.CastedLinear):module.float()
 m.restore_low_dim_params_to_fp32(model)
 blob=(root/'final_model.int8.ptz').read_bytes();blob_sha=hashlib.sha256(blob).hexdigest()
 manifest=json.loads((root/'manifest.json').read_text());assert blob_sha==manifest['model_sha256']
 q=torch.load(io.BytesIO(zlib.decompress(blob)),map_location='cpu',weights_only=True);model.load_state_dict(m.dequantize_state_dict_int8(q),strict=True)
 for param in model.parameters():param.requires_grad_(False)
 before=state_hash(model);compiled=torch.compile(model,dynamic=False,fullgraph=True)
 tokens=m.load_data_shard(root/'fineweb_val_000000.bin').long();n=len(tokens)-1
 sp=spm.SentencePieceProcessor(model_file=str(root/'fineweb_1024_bpe.model'));base,leading,boundary=m.build_sentencepiece_luts(sp,1024,torch.device('cpu'))
 # Baseline prefix fidelity check, then matched full-coverage H2 comparison.
 results=[score(compiled,tokens,n//1024*1024,1024,rank,world,device,base,leading,boundary),score(compiled,tokens,n,1024,rank,world,device,base,leading,boundary),score(compiled,tokens,n,64,rank,world,device,base,leading,boundary)]
 assert results[1]['original_bytes']==manifest['full_validation_decoded_bytes'],'LUT denominator disagrees with independent full-document decode'
 assert results[1]['target_tokens']==results[2]['target_tokens'] and results[1]['original_bytes']==results[2]['original_bytes']
 after=state_hash(model);assert before==after,'Weights changed during evaluation'
 assert abs(results[0]['val_bpb']-manifest['expected_baseline_bpb'])<2e-6,'Fresh evaluation disagrees with training roundtrip'
 dist.barrier()
 output={'checkpoint':root.name,'model_sha256':blob_sha,'frozen_state_hash_before':before,'frozen_state_hash_after':after,'results':results,'total_script_seconds':time.perf_counter()-SCRIPT_START,'torch':torch.__version__,'cuda':torch.version.cuda,'no_ttt':True,'no_training_data_mounted':True,'network_blocked_by_modal':True}
 if rank==0:(root/'evaluation.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
