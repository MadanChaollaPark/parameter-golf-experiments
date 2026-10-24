import datetime,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def update(**kwargs):
 p=ROOT/'STATUS.json';d=json.loads(p.read_text());d.update(kwargs)
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if (ROOT/'budget.json').exists():d['reserved_usd']=json.loads((ROOT/'budget.json').read_text())['total_reserved_usd']
 p.write_text(json.dumps(d,indent=2)+'\n')
