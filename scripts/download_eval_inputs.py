"""Download only pinned public evaluation fixtures before starting offline evaluation."""
import argparse,hashlib,json,os,pathlib,shutil
os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from huggingface_hub import hf_hub_download
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();directory=pathlib.Path(a.directory).resolve()
manifest=json.loads((directory/'manifest.json').read_text());revision='a85b0e6035c3c94bc23685a07c81a8f3bf89db80';assert manifest['data_revision']==revision
paths={'fineweb_val_000000.bin':'datasets/datasets/fineweb10B_sp1024/fineweb_val_000000.bin','fineweb_1024_bpe.model':'datasets/tokenizers/fineweb_1024_bpe.model'}
for name,remote in paths.items():
 source=pathlib.Path(hf_hub_download('willdepueoai/parameter-golf',remote,repo_type='dataset',revision=revision,cache_dir=ROOT/'.cache/eval-fixtures'))
 sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==manifest['file_sha256'][name]
 shutil.copyfile(source,directory/name);print('Verified',name,sha)
print('Fixtures ready. No training shards downloaded. Evaluation can now run offline.')
