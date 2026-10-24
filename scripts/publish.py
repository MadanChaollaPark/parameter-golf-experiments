"""Publish this sanitized worktree only; never submit to the competition."""
import json, os, pathlib, subprocess, urllib.request, urllib.error, re
ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTROL=pathlib.Path('/home/azureuser/parameter-golf-control')
def api(path,method='GET',data=None):
    req=urllib.request.Request('https://api.github.com/'+path,method=method,
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'User-Agent':'parameter-golf-research','Accept':'application/vnd.github+json'},
        data=None if data is None else json.dumps(data).encode())
    with urllib.request.urlopen(req,timeout=40) as r:return json.load(r)
def publish():
    os.chdir(ROOT)
    # Scan every publishable file before staging. Never follow symlinks or stage private inputs.
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name not in ['.budget.lock'] and not p.name.endswith('.tmp')]
    token=os.environ['GH_TOKEN'].encode()
    for p in files:
        assert not p.is_symlink(),str(p)
        assert 'private' not in p.parts and p.stat().st_size<95_000_000,str(p)
        b=p.read_bytes()
        assert token not in b and not re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----\n', b),'Secret detected'
        assert not re.search(rb'(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,})', b),'Credential-shaped content'
    slug='MadanChaollaPark/parameter-golf-experiments'
    try:r=api('repos/'+slug)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        r=api('user/repos','POST',{'name':'parameter-golf-experiments','private':False,'description':'Independent controlled Parameter Golf research; results and limitations, no competition submission'})
    assert r['private'] is False
    subprocess.run(['git','add','--all'],check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode
    if changed:subprocess.run(['git','commit','-m','Record controlled campaign evidence and reproducible artifacts'],check=True)
    subprocess.run(['git','branch','-M','main'],check=True)
    url='https://github.com/'+slug+'.git'
    rem=subprocess.run(['git','remote','get-url','origin'],capture_output=True,text=True)
    if rem.returncode:subprocess.run(['git','remote','add','origin',url],check=True)
    else:assert rem.stdout.strip()==url
    ask=CONTROL/'tools'/'git-askpass-pg.py'
    ask.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "Username" in sys.argv[1] else os.environ["GH_TOKEN"])\n')
    ask.chmod(0o700)
    env=dict(os.environ,GIT_ASKPASS=str(ask),GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git','push','-u','origin','main'],env=env,check=True,timeout=180)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=api('repos/'+slug+'/commits/main')['sha'];assert head==remote
    return {'public':True,'url':'https://github.com/'+slug,'verified_commit':head}
if __name__=='__main__':print(json.dumps(publish()))
