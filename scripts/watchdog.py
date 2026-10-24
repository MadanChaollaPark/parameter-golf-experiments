"""Independent stop/deadline fallback. Only touches campaign app IDs."""
import datetime,json,os,pathlib,subprocess,time,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]; os.chdir(ROOT)
CONTROL=pathlib.Path('/home/azureuser/parameter-golf-control')
PYTHON=str(CONTROL/'venv/bin/python')
start=datetime.datetime.fromisoformat(json.loads((CONTROL/'status.json').read_text())['started_utc']).timestamp()
env=dict(os.environ,MODAL_CONFIG_PATH=str(CONTROL/'private/modal.toml'))
while True:
    if (CONTROL/'FINALIZED').exists():break
    reason='user STOP' if (CONTROL/'STOP').exists() else ('three-hour extension wrap-up deadline' if time.time()>start+3*3600 else '')
    if not reason:time.sleep(10);continue
    (CONTROL/'STOP_JOBS').write_text(reason)
    for attempt in range(3):
        jobs=json.loads((ROOT/'jobs.json').read_text()) if (ROOT/'jobs.json').exists() else []
        for j in jobs:
            if j.get('app_id'):
                subprocess.run([PYTHON,'-m','modal','app','stop',j['app_id']],env=env,capture_output=True,timeout=40)
        time.sleep(2)
    if not (ROOT/'REPORT.md').exists():
        (ROOT/'REPORT.md').write_text('# Interrupted campaign\n\nStopped by '+reason+'. No official result or target achievement is claimed. See STATUS.json, results/, jobs.json and budget.json for durable evidence. Any unvalidated artifact is provisional. Reservations include failures and are not refunded.\n')
    status=json.loads((ROOT/'STATUS.json').read_text());status.update(phase='stopped-finalizing',stop_reason=reason)
    (ROOT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        from publish import publish
        result=publish();(CONTROL/'fallback-publication.json').write_text(json.dumps(result,indent=2))
    except Exception as e:
        (CONTROL/'fallback-error.txt').write_text(type(e).__name__+': '+str(e))
    break
