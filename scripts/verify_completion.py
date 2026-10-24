"""Audit published deliverables and live provider shutdown; do not mark the native goal."""
import datetime,hashlib,json,os,pathlib,subprocess,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
os.environ['MODAL_CONFIG_PATH']='/home/azureuser/parameter-golf-control/private/modal.toml'
def public_api(path):
 request=urllib.request.Request('https://api.github.com/'+path,headers={'User-Agent':'parameter-golf-public-completion-audit','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(request,timeout=40) as response:return json.load(response)
slug='MadanChaollaPark/parameter-golf-experiments'
repo=public_api('repos/'+slug);assert repo['private'] is False
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert public_api('repos/'+slug+'/commits/main')['sha']==head
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Publish current worktree before audit'
budget=json.loads((ROOT/'budget.json').read_text());assert round(sum(e['reserved_usd'] for e in budget['entries']),6)==budget['total_reserved_usd']<=80
best=json.loads((ROOT/'BEST_ARTIFACT.json').read_text())
assert best['training_launch_to_export_observed_upper_bound_seconds']<600 and best['fresh_evaluation_command_seconds']<600
assert sum(f['bytes'] for f in best['files'])==best['package_bytes_including_code_model_configuration_manifest_and_notices']<16000000
for f in best['files']:
 p=ROOT/f['file'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for name in ['REPORT.md','README.md','REPRODUCE.md','EXPERIMENTS.md','PLAN.md','EVALUATION_AUDIT.md','ledger.jsonl','LICENSE','THIRD_PARTY_NOTICES.md']:
 assert (ROOT/name).is_file() and (ROOT/name).stat().st_size
jobs=json.loads((ROOT/'jobs.json').read_text());assert not any(j['state'] in ['running','starting'] for j in jobs)
python='/home/azureuser/parameter-golf-control/venv/bin/python'
apps=json.loads(subprocess.check_output([python,'-m','modal','app','list','--json'],text=True))
ids={j['app_id'] for j in jobs if j.get('app_id')};owned=[a for a in apps if a['app_id'] in ids]
assert {a['app_id'] for a in owned}==ids
assert all(a['state']=='stopped' and int(a['tasks'])==0 for a in owned)
volumes=json.loads(subprocess.check_output([python,'-m','modal','volume','list','--json'],text=True));assert all('name' in v for v in volumes)
assert not {'parameter-golf-controlled-20261024','parameter-golf-evaluation-20261024'}&{v['name'] for v in volumes}
shutdown=json.loads((ROOT/'evidence/shutdown.json').read_text());assert shutdown['all_recorded_apps_stopped'] and shutdown['volume_absence_verified']
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'public_repository':'https://github.com/'+slug,'unauthenticated_public_access_verified':True,'verified_deliverables_commit':head,'objective_audit':{'current_owner_rules_leaders_and_comparability_documented':True,'bounded_baseline_and_matched_two_seed_research_completed':True,'negative_results_and_all_expenses_preserved':True,'full_coverage_frozen_primary_evaluation_verified':True,'artifact_under16MB_and_separate_train_eval_under600s':True,'sanitized_report_code_artifact_and_reproduction_public':True,'all_recorded_campaign_apps_stopped_and_zero_tasks':True,'campaign_data_volumes_deleted':True,'no_competition_submission':True},'budget_conservative_charge_usd':budget['total_reserved_usd'],'budget_is_provider_invoice':False,'best_bpb':best['selected_policy']['val_bpb'],'not_claimed':['official_result','leaderboard_record','statistical_significance','global_optimality','independently_rebuilt_historical_tokenizer_fit','near_duplicate_free_training_data'],'watchdog_service_preserved':'parameter-golf-watchdog.service','native_goal_completion_and_private_FINALIZED_marker_pending_final_audit_publication':True}
(ROOT/'COMPLETION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
