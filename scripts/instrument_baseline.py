"""Recreate timing/correctness-only patch of pinned upstream baseline."""
from pathlib import Path
s=Path('vendor/train_gpt.py').read_text()
s=s.replace('    args = Hyperparameters()\n','    args = Hyperparameters()\n    campaign_start = time.perf_counter()\n',1)
s=s.replace('    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the', '''    log0(f"campaign_pre_warmup_seconds:{time.perf_counter()-campaign_start:.6f}")
    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the''')
s=s.replace('    training_time_ms = 0.0\n','    log0(f"campaign_setup_compile_warmup_seconds:{time.perf_counter()-campaign_start:.6f}")\n    training_time_ms = 0.0\n',1)
s=s.replace('        if should_validate:\n','        if should_validate:\n            if last_step:\n                log0(f"campaign_to_train_end_seconds:{time.perf_counter()-campaign_start:.6f}")\n',1)
s=s.replace('    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())','    campaign_export_start = time.perf_counter()\n    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())',1)
s=s.replace('    t_qeval = time.perf_counter()','    log0(f"campaign_export_seconds:{time.perf_counter()-campaign_export_start:.6f}")\n    log0(f"campaign_to_export_end_seconds:{time.perf_counter()-campaign_start:.6f}")\n    t_qeval = time.perf_counter()',1)
s=s.replace('    if distributed:\n        dist.destroy_process_group()','    log0(f"campaign_total_seconds:{time.perf_counter()-campaign_start:.6f}")\n    if distributed:\n        dist.destroy_process_group()',1)
s=s.replace('        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms', '        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms\n        # Export a clearly labeled partial pilot rather than lose all evidence to the job timeout.\n        if step % 100 == 0 and time.perf_counter() - campaign_start >= float(os.environ.get("CAMPAIGN_TOTAL_SECONDS", "650")):\n            reached_cap = True\n            log0("campaign_safety_stop: partial run; not a completed matched fixed-step experiment")')
s=s.replace('        if distributed and max_wallclock_ms is not None:', '        if distributed and (max_wallclock_ms is not None or step % 100 == 0):')
Path('train_gpt.py').write_text(s)
