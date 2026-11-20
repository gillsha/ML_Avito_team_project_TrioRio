"""Prepare paired Original/Risk prompts and run pinned MLX inference."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GILL = ROOT.parent / 'a08_gill'
sys.path.insert(0, str(GILL))
import rank_a08 as shared

CUTOFF = shared.CUTOFF


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def latest_profiles(data_dir, users):
    """Select latest source row strictly before cutoff; no snapshot fallback."""
    result, late_rows = {}, []
    with (data_dir / 'customer_information.csv').open(encoding='utf-8-sig', newline='') as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            uid = row['customerID']
            if uid not in users:
                continue
            if row['timestamp'][:10] < CUTOFF:
                old = result.get(uid)
                if old is None or row['timestamp'] > old['row']['timestamp']:
                    result[uid] = {'line': line, 'row': row}
            else:
                late_rows.append({'user_id': uid, 'line': line, 'timestamp': row['timestamp'],
                                  'riskLevel': row.get('riskLevel', '')})
    return result, late_rows


def profile_value(value, timestamp, exists):
    if not exists or timestamp[:10] == '2000-01-01' or timestamp[:10] >= CUTOFF:
        return {'value': 'unknown', 'status': 'unverified'}
    if value in ('', 'Not_Available', 'nan'):
        return {'value': 'unknown', 'status': 'missing'}
    if value.startswith('Predicted_'):
        return {'value': 'unknown', 'status': 'predicted'}
    return {'value': value, 'status': 'observed'}


def safe_profile(row):
    """Mask every value unless its source row has a documented pre-cutoff timestamp."""
    if not row or row['timestamp'][:10] >= CUTOFF or row['timestamp'][:10] == '2000-01-01':
        return {key: {'value': 'unknown', 'status': 'unverified'}
                for key in ('customerType', 'riskLevel', 'investmentCapacity')}
    return {key: profile_value(row.get(key, ''), row['timestamp'], True)
            for key in ('customerType', 'riskLevel', 'investmentCapacity')}


def paired_prompts(original_instruction, payload, risk_instruction):
    """Append one condition-specific instruction; payload bytes are shared exactly."""
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return (original_instruction + '\n' + payload_text,
            original_instruction + ' ' + risk_instruction.strip() + '\n' + payload_text)


def prepare(args):
    args.starter_dir = GILL / 'course' / 'far-trans'
    args.cutoff, args.horizon_days, args.max_users = CUTOFF, 30, 100
    args.candidate_count = 50
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = shared.data_adapter.load(args)
    base_dir = args.baseline_dir
    protocol = json.loads((base_dir / 'protocol.json').read_text())
    base_hash = protocol.pop('protocol_hash')
    assert base_hash == digest(protocol)
    assert protocol['users'] == manifest['users'] and len(manifest['users']) == 100
    assert protocol['cutoff'] == CUTOFF and protocol['exclude_seen'] is False
    assert protocol['candidate_count'] == 50 and protocol['k'] == 10
    assert protocol['source_sha256'] == manifest['data_sha256']
    profiles, late_rows = latest_profiles(args.data_dir, set(manifest['users']))
    requests, candidates = [], {}
    for uid, context in contexts.items():
        ids = starter.order({isin: pop[isin] for isin in context['eligible_asset_ids']})[:50]
        assert len(ids) == 50 and ids[:10] == baseline[uid]['popularity']
        candidates[uid] = ids
        # Use the exact original prompt-construction mechanism, then safely mask profile.
        original = shared.ranker.build_prompt(
            dict(context, candidate_assets=[{'id': isin} for isin in ids]), assets,
            history[uid], events[uid], history_events=shared.HISTORY_EVENTS)
        instruction, raw_payload = original.split('\n', 1)
        payload = json.loads(raw_payload)
        row = profiles.get(uid, {}).get('row')
        payload['profile_as_of_cutoff'] = safe_profile(row)
        payload = {'candidate_assets': payload.pop('candidate_assets'), **payload}
        original_prompt, risk_prompt = paired_prompts(
            instruction, payload, (ROOT / 'risk_instruction.txt').read_text())
        assert original_prompt.split('\n', 1)[1].encode() == risk_prompt.split('\n', 1)[1].encode()
        requests.append({'user_id': uid, 'candidate_ids': ids,
                         'original': original_prompt, 'risk': risk_prompt})
    protocol.update(ranker_candidate_source='popularity', ranker_candidates_sha256=digest(candidates))
    protocol['protocol_hash'] = digest(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared.write_json(args.output_dir / 'protocol.json', protocol)
    shared.write_jsonl(args.output_dir / 'prompts.jsonl', requests)
    audit_profiles(manifest['users'], profiles, late_rows, truth, contexts, args.output_dir)
    shared.write_json(args.output_dir / 'paired_manifest.json', {
        'protocol_hash': protocol['protocol_hash'],
        'baseline_protocol_hash': base_hash,
        'baseline_metrics_sha256': shared.file_hash(base_dir / 'metrics.json'),
        'profile_policy': 'source timestamp strictly before cutoff; placeholder/missing/predicted/unverified masked',
        'risk_instruction_sha256': shared.file_hash(ROOT / 'risk_instruction.txt'),
        'candidate_sha256': protocol['ranker_candidates_sha256']})
    print(f"Prepared {len(requests)} paired users; candidate SHA-256={digest(candidates)}; protocol={protocol['protocol_hash']}")


def audit_profiles(users, profiles, late_rows, truth, contexts, outdir):
    from collections import Counter
    future = {uid for uid in users
              if truth.get(uid, set()) & set(contexts[uid]['eligible_asset_ids'])}
    vals = {uid: safe_profile(profiles[uid]['row'] if uid in profiles else None) for uid in users}
    counts = Counter(v['status'] for p in vals.values() for v in p.values())
    risk_known = {u for u, p in vals.items() if p['riskLevel']['status'] == 'observed'}
    aggressive = {u for u in risk_known if vals[u]['riskLevel']['value'] == 'Aggressive'}
    professional = {u for u, p in vals.items() if p['customerType']['status'] == 'observed' and p['customerType']['value'] == 'Professional'}
    subgroup = aggressive | professional
    result = {
        'users': len(users), 'confirmed_observed_riskLevel_users': len(risk_known),
        'riskLevel_Aggressive_users': len(aggressive), 'customerType_Professional_users': len(professional),
        'union_without_double_count': len(subgroup),
        'subgroup_users_with_future_buys': len(subgroup & future), 'future_buyer_count_all': len(future),
        'profile_field_status_counts': dict(counts),
        'riskLevel_missing_values': sum(p['riskLevel']['status'] == 'missing' for p in vals.values()),
        'riskLevel_predicted_values': sum(p['riskLevel']['status'] == 'predicted' for p in vals.values()),
        'riskLevel_unconfirmed_values': sum(p['riskLevel']['status'] == 'unverified' for p in vals.values()),
        'riskLevel_late_values_excluded': sum(bool(r['riskLevel']) for r in late_rows),
        'users_with_late_profile_rows': len({r['user_id'] for r in late_rows}),
        'late_profile_rows_excluded': len(late_rows),
        'users_without_pre_cutoff_source_row': sum(uid not in profiles for uid in users),
        'riskLevel_status_counts': dict(Counter(p['riskLevel']['status'] for p in vals.values())),
        'customerType_status_counts': dict(Counter(p['customerType']['status'] for p in vals.values())),
        'explanation': 'Availability is proven only by a non-placeholder source-row timestamp strictly before cutoff (README defines it as last update). lastQuestionnaireDate alone is not used; snapshots at/after cutoff are excluded.'}
    shared.write_json(outdir / 'profile_audit.json', result)
    provenance = []
    for uid in users:
        item = profiles.get(uid)
        row = item['row'] if item else None
        provenance.append({'user_id': uid, 'source_line': item['line'] if item else None,
                           'source_timestamp': row['timestamp'] if row else None,
                           'lastQuestionnaireDate_not_used_for_availability': row.get('lastQuestionnaireDate') if row else None,
                           'safe_profile': vals[uid]})
    shared.write_jsonl(outdir / 'profile_provenance.jsonl', provenance)
    shared.write_jsonl(outdir / 'late_profile_rows_excluded.jsonl', late_rows)
    return result


def parse_or_fallback(request, raw):
    try:
        ids, repaired = shared.ranker.parse_ranking(raw, set(request['candidate_ids']))
        return ids, repaired, False, None
    except (ValueError, TypeError) as exc:
        return request['candidate_ids'][:10], False, True, str(exc)


def common_token_prefix(prompts):
    if not prompts or any(not p for p in prompts):
        raise ValueError('Cannot cache an empty prompt set')
    prefix = []
    shortest = min(map(len, prompts))
    for index, column in enumerate(zip(*prompts)):
        if index == shortest - 1 or len(set(column)) != 1:
            break
        prefix.append(column[0])
    if not prefix:
        raise ValueError('Expected a non-empty common prompt prefix')
    return prefix


def prepare_cached_inputs(model, tokenizer, requests):
    """Prefill each variant's identical token prefix once, like rank_a08.py."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    tokenized, prefix_lengths, caches = {}, {}, {}
    for variant in ('original', 'risk'):
        prompts = []
        for request in requests:
            tokens = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': request[variant]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            tokenized[request['user_id'], variant] = tokens
            prompts.append(tokens)
        prefix = common_token_prefix(prompts)
        cache = make_prompt_cache(model)
        for start in range(0, len(prefix), 1024):
            model(mx.array([prefix[start:start + 1024]]), cache=cache)
            mx.eval([layer.state for layer in cache])
            mx.clear_cache()
        prefix_lengths[variant] = len(prefix)
        caches[variant] = cache
    return tokenized, prefix_lengths, caches


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class InferenceInfrastructureError(RuntimeError):
    pass


def call_generator(generate):
    """Wrap model/runtime failures; they are never converted to ranking fallback."""
    try:
        return generate()
    except Exception as exc:
        raise InferenceInfrastructureError(f'{type(exc).__name__}: {exc}') from exc


def journal_path(run_dir, user_id, variant):
    key = hashlib.sha256(user_id.encode('utf-8')).hexdigest()
    return run_dir / 'journal' / variant / f'{key}.json'


def run_location(base_dir, selected_ids, frozen_users):
    full = selected_ids == frozen_users
    return full, base_dir / ('inference' if full else 'pilot')


def compatible_answer(path, expected, candidate_ids):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    keys = ('user_id', 'variant', 'protocol_hash', 'prompt_sha256', 'settings_hash', 'candidate_ids')
    mismatches = [key for key in keys if record.get(key) != expected[key]]
    if mismatches:
        raise ValueError(f'Incompatible saved answer {path.name}; mismatched: {", ".join(mismatches)}')
    ranking = record.get('ranked_ids')
    if (not isinstance(ranking, list) or len(ranking) != 10
            or any(type(item) is not str for item in ranking) or len(set(ranking)) != 10
            or not set(ranking) <= set(candidate_ids)):
        raise ValueError(f'Invalid saved ranking in {path.name}')
    if record.get('fallback') and (ranking != candidate_ids[:10] or not record.get('error')):
        raise ValueError(f'Invalid fallback record in {path.name}')
    if not record.get('fallback'):
        raw = record.get('raw_response')
        parsed, _ = shared.ranker.parse_ranking(raw, set(candidate_ids))
        if parsed != ranking:
            raise ValueError(f'Saved ranking does not match raw response in {path.name}')
    return record


def save_infrastructure_error(run_dir, user_id, variant, stage, exc):
    key = hashlib.sha256(user_id.encode('utf-8')).hexdigest()
    atomic_json(run_dir / 'infrastructure_errors' / variant / f'{key}.json', {
        'user_id': user_id, 'variant': variant, 'stage': stage,
        'error_type': type(exc).__name__, 'error': str(exc), 'fallback': False})


def publish_results(run_dir, variant, selected, protocol_hash, settings_hash, full_users,
                    require_full=False):
    selected_ids = [request['user_id'] for request in selected]
    is_full = selected_ids == full_users
    if require_full and not is_full:
        raise ValueError('Full publication requires the exact frozen user order')
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError('Duplicate users in publication cohort')
    records = []
    for request in selected:
        uid = request['user_id']
        expected = {
            'user_id': uid, 'variant': variant, 'protocol_hash': protocol_hash,
            'prompt_sha256': hashlib.sha256(request[variant].encode()).hexdigest(),
            'settings_hash': settings_hash, 'candidate_ids': request['candidate_ids']}
        record = compatible_answer(journal_path(run_dir, uid, variant), expected,
                                   request['candidate_ids'])
        if record is None:
            raise ValueError(f'Cannot publish incomplete {variant} results: {uid} missing')
        records.append(record)
    if is_full and (len(records) != len(full_users) or
                    [r['user_id'] for r in records] != full_users):
        raise ValueError('Refusing to publish a partial cohort as full')
    atomic_jsonl(run_dir / f'{variant}_responses.jsonl', records)
    return records


def infer(args):
    requests = shared.read_jsonl(args.output_dir / 'prompts.jsonl')
    protocol = json.loads((args.output_dir / 'protocol.json').read_text())
    frozen_users = protocol['users']
    if len(requests) != 100 or [r['user_id'] for r in requests] != frozen_users:
        raise ValueError('Prompts must contain the exact frozen 100-user cohort and order')
    if args.user_ids:
        if len(args.user_ids) != len(set(args.user_ids)):
            raise ValueError('--user-ids must be unique')
        by_id = {request['user_id']: request for request in requests}
        unknown = set(args.user_ids) - set(by_id)
        if unknown:
            raise ValueError(f'--user-ids must be selected from the frozen cohort; unknown count={len(unknown)}')
        selected = [by_id[uid] for uid in args.user_ids]
    else:
        selected = requests[:args.pilot_users] if args.pilot_users else requests
    if not selected:
        raise ValueError('Inference selection cannot be empty')
    selected_ids = [r['user_id'] for r in selected]
    full_run, run_dir = run_location(args.output_dir, selected_ids, frozen_users)
    # Only the exact frozen cohort in order can write to the full-run directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / 'selected_users.json', {
        'user_ids': selected_ids, 'selected_count': len(selected),
        'selection': 'exact frozen cohort' if full_run else 'partial cohort/pilot'})

    model_files = {name: shared.file_hash(args.model_dir / name) for name in
                   ('config.json', 'tokenizer.json', 'tokenizer_config.json', *shared.WEIGHT_SHA256)}
    for shard, expected in shared.WEIGHT_SHA256.items():
        if model_files[shard] != expected:
            raise ValueError(f'Model shard {shard} does not match pinned revision')
    import mlx.core as mx
    if 'gpu' not in str(mx.default_device()).lower():
        raise RuntimeError(f'Expected Apple GPU/Metal, got {mx.default_device()}')
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler
    try:
        model, tokenizer = shared.open_model(args.model_dir)
    except Exception as exc:
        atomic_json(run_dir / 'infrastructure_errors' / 'model_load.json', {
            'stage': 'model_load', 'error_type': type(exc).__name__, 'error': str(exc),
            'fallback': False})
        raise InferenceInfrastructureError(f'Model load failed: {type(exc).__name__}: {exc}') from exc
    tokenizer.eos_token_ids = [151645, 151643]
    try:
        tokenized, prefix_lengths, caches = prepare_cached_inputs(model, tokenizer, requests)
    except Exception as exc:
        atomic_json(run_dir / 'infrastructure_errors' / 'prefix_cache.json', {
            'stage': 'prefix_cache', 'error_type': type(exc).__name__, 'error': str(exc),
            'fallback': False})
        raise InferenceInfrastructureError(f'Prefix cache preparation failed: {type(exc).__name__}: {exc}') from exc

    variants = ('original', 'risk')
    settings = {
        'model_id': shared.MODEL, 'revision': shared.REVISION,
        'model_files_sha256': model_files, 'device': str(mx.default_device()),
        'device_backend': 'Apple GPU (MLX Metal)', 'dtype': 'bfloat16 unquantized',
        'temperature': 0.0, 'thinking': False, 'max_new_tokens': 512,
        'context_limit': 8192, 'protocol_hash': protocol['protocol_hash'],
        'eos_token_ids': [151645, 151643], 'history_events': shared.HISTORY_EVENTS,
        'system': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers')},
        'python': sys.version, 'hardware': args.hardware,
        'prompts_sha256': shared.file_hash(args.output_dir / 'prompts.jsonl'),
        'risk_instruction_sha256': shared.file_hash(ROOT / 'risk_instruction.txt'),
        'prefix_cache': 'shared token prefix prefilled once per variant; remainder generated per answer',
        'prefix_tokens': prefix_lengths,
    }
    settings_hash = digest(settings)
    settings_path = run_dir / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing inference settings are incompatible; refusing resume')
    if not settings_path.exists():
        atomic_json(settings_path, settings)

    for request in selected:
        for variant in variants:
            uid = request['user_id']
            expected = {
                'user_id': uid, 'variant': variant,
                'protocol_hash': protocol['protocol_hash'],
                'prompt_sha256': hashlib.sha256(request[variant].encode()).hexdigest(),
                'settings_hash': settings_hash,
                'candidate_ids': request['candidate_ids'],
            }
            path = journal_path(run_dir, uid, variant)
            if compatible_answer(path, expected, request['candidate_ids']) is not None:
                print(f'resumed {variant} {selected_ids.index(uid) + 1}/{len(selected)}', flush=True)
                continue
            tokens = tokenized[uid, variant]
            if len(tokens) + settings['max_new_tokens'] > settings['context_limit']:
                exc = ValueError(f'Prompt plus output budget exceeds context limit ({len(tokens)} tokens)')
                save_infrastructure_error(run_dir, uid, variant, 'context_check', exc)
                raise InferenceInfrastructureError(str(exc)) from exc
            suffix = tokens[prefix_lengths[variant]:]
            started = time.perf_counter()
            try:
                def generate_one():
                    response = batch_generate(
                        model, tokenizer, [suffix], prompt_caches=[caches[variant]],
                        max_tokens=settings['max_new_tokens'], sampler=make_sampler(temp=0.0))
                    if len(response.texts) != 1 or type(response.texts[0]) is not str:
                        raise RuntimeError('MLX batch_generate returned an invalid response shape')
                    return response, int(response.stats.generation_tokens)
                result, output_tokens = call_generator(generate_one)
            except InferenceInfrastructureError as exc:
                save_infrastructure_error(run_dir, uid, variant, 'generation', exc)
                raise
            generation_seconds = time.perf_counter() - started
            raw = result.texts[0]
            # Only parsing/validation errors are user-level invalid responses and fall back.
            ranked, repaired, fallback, error = parse_or_fallback(request, raw)
            payload = json.loads(request[variant].split('\n', 1)[1])
            record = {
                **expected, 'stage': 'ranker', 'ranked_ids': ranked,
                'raw_response': raw, 'fallback': fallback, 'error': error,
                'parse_repaired': repaired, 'prompt': request[variant],
                'profile_as_of_cutoff': payload['profile_as_of_cutoff'],
                'input_tokens': len(tokens), 'output_tokens': output_tokens,
                'generation_seconds': generation_seconds,
            }
            atomic_json(path, record)
            print(f'saved {variant} {selected_ids.index(uid) + 1}/{len(selected)} '
                  f'fallback={fallback}', flush=True)

    published = {}
    for variant in variants:
        published[variant] = publish_results(
            run_dir, variant, selected, protocol['protocol_hash'], settings_hash,
            frozen_users, require_full=full_run)
    hashes = {path.name: shared.file_hash(path) for path in
              (args.output_dir / 'protocol.json', args.output_dir / 'prompts.jsonl',
               args.output_dir / 'profile_audit.json', run_dir / 'settings.json',
               args.output_dir / 'paired_manifest.json',
               args.output_dir / 'profile_provenance.jsonl',
               args.output_dir / 'late_profile_rows_excluded.jsonl',
               run_dir / 'selected_users.json',
               run_dir / 'original_responses.jsonl', run_dir / 'risk_responses.jsonl')}
    shared.write_json(run_dir / 'artifacts_sha256.json', hashes)
    for v in variants:
        records = published[v]
        print(f"{v}: valid={sum(not r['fallback'] for r in records)}/{len(records)}, "
              f"fallback={sum(r['fallback'] for r in records)}; output={run_dir}")


def analyze(args):
    """Summarize unchanged evaluator outputs overall and in the pre-cutoff subgroup."""
    profiles = shared.read_jsonl(args.paired_dir / 'profile_provenance.jsonl')
    group = {r['user_id'] for r in profiles
             if (r['safe_profile']['riskLevel']['status'] == 'observed'
                 and r['safe_profile']['riskLevel']['value'] == 'Aggressive')
             or (r['safe_profile']['customerType']['status'] == 'observed'
                 and r['safe_profile']['customerType']['value'] == 'Professional')}
    dirs = {'baseline': args.baseline_dir, 'original': args.original_dir, 'risk': args.risk_dir}
    protocols = {name: json.loads((path / 'protocol.json').read_text()) for name, path in dirs.items()}
    shared_fields = {k: v for k, v in protocols['baseline'].items() if k != 'protocol_hash'}
    for name in ('original', 'risk'):
        protocol = protocols[name]
        reduced = {k: v for k, v in protocol.items()
                   if k not in {'protocol_hash', 'ranker_candidate_source', 'ranker_candidates_sha256'}}
        if reduced != shared_fields:
            raise ValueError(f'{name} protocol differs from the frozen baseline protocol')
        if protocol['protocol_hash'] != json.loads((args.paired_dir / 'protocol.json').read_text())['protocol_hash']:
            raise ValueError(f'{name} ranker protocol hash does not match paired prompts')
    summary = {'protocol_hash': protocols['original']['protocol_hash'],
               'users': len(protocols['baseline']['users']), 'subgroup_users': len(group),
               'methods': {}, 'risk_instruction_sha256': shared.file_hash(ROOT / 'risk_instruction.txt')}
    users = set(protocols['baseline']['users'])
    evaluation_rows = {}
    for name, directory in dirs.items():
        with (directory / 'per_user.csv').open(encoding='utf-8', newline='') as f:
            records = list(csv.DictReader(f))
        by_method = {}
        for row in records:
            by_method.setdefault(row['method'], {})[row['user_id']] = row
        if not {'popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'} <= set(by_method):
            raise ValueError(f'{name} is missing a required classical baseline')
        if name != 'baseline' and 'custom' not in by_method:
            raise ValueError(f'{name} has no evaluated custom ranker')
        for method, rows in by_method.items():
            if set(rows) != users:
                raise ValueError(f'{name}/{method} does not contain every frozen cohort user')
            evaluation_rows[name, method] = rows
            for subset_name, subset in (('all', users), ('risk_or_professional', group)):
                selected = [rows[u] for u in subset]
                nonempty = [r for r in selected if int(r['relevant_count']) > 0]
                label = f'{name}_custom' if method == 'custom' else method
                block = summary['methods'].setdefault(label, {})
                block[f'{name}_{subset_name}'] = {
                    'users': len(selected), 'nonempty_truth_users': len(nonempty),
                    'ndcg_at_10': math.fsum(float(r['ndcg_at_10']) for r in selected) / len(selected),
                    'ndcg_at_10_nonempty': (math.fsum(float(r['ndcg_at_10']) for r in nonempty) / len(nonempty)
                     if nonempty else None)}
    if evaluation_rows['original', 'popularity'] != evaluation_rows['risk', 'popularity']:
        raise ValueError('Popularity scores differ between Original and Risk evaluator outputs')

    def mean_for(rows, subset):
        chosen = [rows[uid] for uid in subset]
        return math.fsum(float(r['ndcg_at_10']) for r in chosen) / len(chosen) if chosen else None

    comparisons = {}
    for subset_name, subset in (('all', users), ('risk_or_professional', group)):
        risk_rows = evaluation_rows['risk', 'custom']
        original_rows = evaluation_rows['original', 'custom']
        popular_rows = evaluation_rows['baseline', 'popularity']
        risk_mean, original_mean, popular_mean = (
            mean_for(risk_rows, subset), mean_for(original_rows, subset), mean_for(popular_rows, subset))
        comparisons[subset_name] = {
            'risk_minus_original_ndcg_at_10': risk_mean - original_mean,
            'risk_minus_popularity_ndcg_at_10': risk_mean - popular_mean,
            'risk_ndcg_at_10': risk_mean, 'original_ndcg_at_10': original_mean,
            'popularity_ndcg_at_10': popular_mean, 'users_in_primary_mean': len(subset)}

    inference_dir = getattr(args, 'inference_dir', None) or args.paired_dir / 'inference'
    valid_ids = {}
    fallback_counts = {}
    for variant in ('original', 'risk'):
        records = shared.read_jsonl(inference_dir / f'{variant}_responses.jsonl')
        if len(records) != len(users) or {r['user_id'] for r in records} != users:
            raise ValueError(f'{variant} responses are not the complete frozen cohort')
        valid_ids[variant] = {r['user_id'] for r in records if not r['fallback']}
        fallback_counts[variant] = len(records) - len(valid_ids[variant])
    paired_valid = valid_ids['original'] & valid_ids['risk']
    valid_only = {}
    for subset_name, subset in (('all', users), ('risk_or_professional', group)):
        paired = paired_valid & subset
        risk_valid = valid_ids['risk'] & subset
        risk_paired_score = mean_for(evaluation_rows['risk', 'custom'], paired) if paired else None
        original_paired_score = mean_for(evaluation_rows['original', 'custom'], paired) if paired else None
        risk_valid_score = mean_for(evaluation_rows['risk', 'custom'], risk_valid) if risk_valid else None
        popularity_risk_valid_score = mean_for(evaluation_rows['baseline', 'popularity'], risk_valid) if risk_valid else None
        valid_only[subset_name] = {
            'paired_valid_users': len(paired),
            'risk_ndcg_at_10_paired_valid_only': risk_paired_score,
            'original_ndcg_at_10_paired_valid_only': original_paired_score,
            'risk_minus_original_ndcg_at_10_paired_valid_only': (risk_paired_score - original_paired_score
                                                                  if paired else None),
            'risk_valid_users': len(risk_valid),
            'risk_ndcg_at_10_risk_valid_only': risk_valid_score,
            'popularity_ndcg_at_10_risk_valid_only': popularity_risk_valid_score,
            'risk_minus_popularity_ndcg_at_10_risk_valid_only': (risk_valid_score - popularity_risk_valid_score
                                                                  if risk_valid else None),
        }
    summary['comparisons'] = comparisons
    summary['fallback_counts_kept_in_primary_metrics'] = fallback_counts
    summary['valid_only_diagnostic_not_primary'] = valid_only
    shared.write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data-dir', type=Path, required=True)
    prep.add_argument('--baseline-dir', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    inf = sub.add_parser('infer')
    inf.add_argument('--output-dir', type=Path, required=True)
    inf.add_argument('--model-dir', type=Path, required=True)
    inf.add_argument('--hardware', required=True)
    inf.add_argument('--pilot-users', type=int, default=3)
    inf.add_argument('--user-ids', nargs='+',
                     help='Explicit IDs drawn only from the frozen 100-user cohort; overrides --pilot-users')
    ana = sub.add_parser('analyze')
    ana.add_argument('--paired-dir', type=Path, required=True)
    ana.add_argument('--baseline-dir', type=Path, required=True)
    ana.add_argument('--original-dir', type=Path, required=True)
    ana.add_argument('--risk-dir', type=Path, required=True)
    ana.add_argument('--output', type=Path, required=True)
    ana.add_argument('--inference-dir', type=Path,
                     help='Defaults to PAIRED_DIR/inference; used only for fallback diagnostics')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'infer':
        if args.pilot_users < 0 or args.pilot_users > 100:
            p.error('--pilot-users must be between 0 and 100')
        infer(args)
    else:
        analyze(args)

if __name__ == '__main__':
    main()
