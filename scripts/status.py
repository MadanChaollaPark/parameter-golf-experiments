import datetime,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def update(**kwargs):
 p=ROOT/'STATUS.json';d=json.loads(p.read_text());d.update(kwargs)
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if (ROOT/'budget.json').exists():
  budget=json.loads((ROOT/'budget.json').read_text());d['reserved_usd']=budget['total_reserved_usd']
  d['uncommitted_usd']=round(budget['cap_usd']-budget['total_reserved_usd'],6)
  d['protected_final_usd']=max(0,budget['final_holdback_usd']-sum(e['reserved_usd'] for e in budget['entries'] if e.get('phase')=='final'))
  d['cost_basis']='Conservative nonrefunded reservations at full list price, not provider invoice'
 p.write_text(json.dumps(d,indent=2)+'\n')
