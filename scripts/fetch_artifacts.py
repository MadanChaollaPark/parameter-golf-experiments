import hashlib,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.environ['MODAL_CONFIG_PATH']='/home/azureuser/parameter-golf-control/private/modal.toml'
import modal
volume=modal.Volume.from_name('parameter-golf-controlled-20261024')
for job in sys.argv[1:]:
 d=json.loads((ROOT/'results'/f'{job}.json').read_text());target=ROOT/'artifacts'/job;target.mkdir(exist_ok=True,parents=True)
 for f in d['artifacts']:
  dest=target/pathlib.Path(f['path']).name
  with dest.open('wb') as out:
   for b in volume.read_file(f['path']):out.write(b)
  assert dest.stat().st_size==f['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==f['sha256']
  print('verified artifact',job,dest.name,f['bytes'])
