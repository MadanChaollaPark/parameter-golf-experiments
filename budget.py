"""Conservative reservation ledger. All reservations remain charged until audited reconciliation."""
import argparse, datetime, fcntl, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'budget.json'

def reserve(job, dollars, final=False, details=''):
    if not dollars > 0:
        raise ValueError('positive reservation required')
    with (ROOT / '.budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
            'cap_usd': 80.0, 'final_holdback_usd': 20.0, 'entries': [{
                'job': 'controller-setup-storage-egress', 'reserved_usd': 8.0,
                'kind': 'conservative_overhead_allowance',
                'details': 'Azure <=8h, CPU setup, storage/egress. Not provider invoice.'}]}
        if any(x['job'] == job for x in d['entries']):
            raise ValueError('duplicate reservation id')
        charged = sum(x['reserved_usd'] for x in d['entries'])
        limit = d['cap_usd'] - (0 if final else d['final_holdback_usd'])
        if charged + dollars > limit + 1e-8:
            raise ValueError(f'budget denied: {charged} + {dollars} > {limit}')
        if d.get('extension_authorization'):
            frozen=json.loads((ROOT/'extension/evidence/prior-ledger-frozen.json').read_text())
            assert d['entries'][:len(frozen['entries'])]==frozen['entries']
            assert job.startswith('extension-'),'Extension job prefix required'
        d['entries'].append({'job': job, 'reserved_usd': dollars,
            'phase': 'extension' if d.get('extension_authorization') else ('final' if final else 'research'), 'details': details,
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        d['total_reserved_usd'] = round(charged + dollars, 6)
        tmp = LEDGER.with_suffix('.tmp'); tmp.write_text(json.dumps(d,indent=2)+'\n')
        os.replace(tmp, LEDGER)
        return d

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('job'); p.add_argument('dollars',type=float)
    p.add_argument('--final',action='store_true'); p.add_argument('--details',default='')
    a=p.parse_args(); print(json.dumps(reserve(a.job,a.dollars,a.final,a.details),indent=2))
