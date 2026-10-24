"""Fail closed if current files or Git history contain credential-shaped content."""
import datetime,json,os,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
token=os.environ.get('GH_TOKEN','').encode();assert token
patterns=[rb'-----BEGIN [A-Z ]*PRIVATE KEY-----\n',rb'(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,})',rb'(?:ak|as)-[A-Za-z0-9]{24,}',rb'X-Amz-Signature=[0-9a-fA-F]{30,}']
def check(data,identity):
 assert token not in data,f'Exact credential found in {identity}'
 assert not any(re.search(p,data) for p in patterns),f'Credential-shaped content found in {identity}'
files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name!='.budget.lock' and not p.name.endswith('.tmp')]
for p in files:
 assert not p.is_symlink() and 'private' not in p.relative_to(ROOT).parts
 assert p.stat().st_size<95_000_000
 check(p.read_bytes(),str(p.relative_to(ROOT)))
objects=subprocess.check_output(['git','rev-list','--objects','--all'],text=True).splitlines()
metadata=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype)'],input='\n'.join(x.split()[0] for x in objects)+'\n',text=True).splitlines()
blobs=[x.split()[0] for x in metadata if x.endswith(' blob')]
for oid in blobs:check(subprocess.check_output(['git','cat-file','blob',oid]),'Git blob '+oid)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worktree_files_checked':len(files),'history_blobs_checked':len(blobs),'credential_checks_passed':True,'private_paths_excluded':True,'scope':'Exact current GitHub credential, standard GitHub/Modal token shapes, PEM keys, AWS signed URL signatures; current files and all reachable Git blobs. No raw controller transcripts included.'}
(ROOT/'evidence/publication-sanitization.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
