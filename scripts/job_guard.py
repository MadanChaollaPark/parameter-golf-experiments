"""Deadline guardian discovers only this exact campaign app if startup has not returned."""
import json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];CONTROL=pathlib.Path('/home/azureuser/parameter-golf-control')
start=time.monotonic();limit=int(sys.argv[2]);env=dict(os.environ,MODAL_CONFIG_PATH=str(CONTROL/'private/modal.toml'))
while time.monotonic()-start<limit and not (CONTROL/'STOP').exists() and not (CONTROL/'STOP_JOBS').exists():time.sleep(5)
py=str(CONTROL/'venv/bin/python')
r=subprocess.run([py,'-m','modal','app','list','--json'],env=env,capture_output=True,text=True,timeout=40)
for j in json.loads(r.stdout):
 if j['description']==(sys.argv[3] if len(sys.argv)>3 else 'parameter-golf-controlled-20261024') and j['state']!='stopped':
  subprocess.run([py,'-m','modal','app','stop',j['app_id']],env=env,capture_output=True,timeout=40)
