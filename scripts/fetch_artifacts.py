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
 # Preserve the trainer's native log, which embeds the exact executed source.
 native=target/(job+'.txt')
 try:
  with native.open('wb') as out:
   for b in volume.read_file('runs/'+job+'/logs/'+job+'.txt'):out.write(b)
  (target/'native_log_manifest.json').write_text(json.dumps({'file':native.name,'bytes':native.stat().st_size,'sha256':hashlib.sha256(native.read_bytes()).hexdigest()},indent=2)+'\n')
  print('preserved native source-embedded log',job)
 except Exception as e:
  if native.exists() and native.stat().st_size==0:native.unlink()
  print('native log unavailable',job,type(e).__name__)
