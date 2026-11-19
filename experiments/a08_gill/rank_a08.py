"""Build fixed-Popularity prompts and run the original and category-aware variants."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'course' / 'far-trans-llm'))
import data_adapter
import ranker

CUTOFF = '2021-06-01'
CANDIDATE_COUNT = 50
HISTORY_EVENTS = 15
MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
WEIGHT_SHA256 = {
    'model-00001-of-00002.safetensors': '169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5',
    'model-00002-of-00002.safetensors': '912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9',
}
MAX_TOKENS = 512
CONTEXT_LIMIT = 8192
VARIANTS = ('category', 'original')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def prepare(args):
    args.starter_dir = ROOT / 'course' / 'far-trans'
    args.cutoff = CUTOFF
    args.horizon_days = 30
    args.max_users = 100
    args.candidate_count = CANDIDATE_COUNT
    starter, assets, history, _, contexts, baseline, events, pop, manifest = data_adapter.load(args)
    proto = json.loads((args.baseline_dir / 'protocol.json').read_text())
    original_hash = proto.pop('protocol_hash')
    assert original_hash == digest(proto)
    assert proto['users'] == manifest['users'] and len(proto['users']) == 100
    assert proto['eligible_assets'] == sorted(assets)
    assert proto['cutoff'] == CUTOFF and proto['end_exclusive'] == manifest['end_exclusive']
    assert proto['cohort_selection'] == 'warm' and not proto['exclude_seen']
    assert proto['candidate_count'] == CANDIDATE_COUNT and proto['k'] == 10
    assert proto['source_sha256'] == manifest['data_sha256']

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, requests = {}, []
    instruction = (ROOT / 'category_instruction.txt').read_text().strip()
    for uid, context in contexts.items():
        ids = starter.order({isin: pop[isin] for isin in context['eligible_asset_ids']})[:CANDIDATE_COUNT]
        assert len(ids) == CANDIDATE_COUNT and ids[:10] == baseline[uid]['popularity']
        candidates[uid] = ids
        ranker_context = dict(context, candidate_assets=[{'id': isin} for isin in ids])
        original = ranker.build_prompt(
            ranker_context, assets, history[uid], events[uid], history_events=HISTORY_EVENTS)
        prefix, payload = original.split('\n', 1)
        fields = json.loads(payload)
        # Popularity supplies the same metadata prefix for all warm users.
        # Keep the fields and values, but serialize that common prefix first.
        fields = {'candidate_assets': fields.pop('candidate_assets'), **fields}
        payload = json.dumps(fields, ensure_ascii=False, separators=(',', ':'))
        requests.append({'user_id': uid, 'candidate_ids': ids, 'original': prefix + '\n' + payload,
                         'category': prefix + ' ' + instruction + '\n' + payload})
    proto.update(ranker_candidate_source='popularity', ranker_candidates_sha256=digest(candidates))
    proto['protocol_hash'] = digest(proto)
    write_json(args.output_dir / 'expected_ranker_protocol.json', proto)
    write_jsonl(args.output_dir / 'prompts.jsonl', requests)
    print(f'Prepared {len(requests)} users, candidate SHA-256 {digest(candidates)}', flush=True)


def open_model(model_dir):
    from mlx_lm import load
    return load(str(model_dir), tokenizer_config={'trust_remote_code': False})


def prepare_model_inputs(model, tokenizer, requests):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    tokenized, prefix_lengths, caches = {}, {}, {}
    for variant in VARIANTS:
        prompts = []
        for request in requests:
            tokens = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': request[variant]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            prompts.append(tokens)
            tokenized[request['user_id'], variant] = tokens
        prefix = []
        shortest = min(map(len, prompts))
        for index, column in enumerate(zip(*prompts)):
            if index == shortest - 1 or len(set(column)) != 1:
                break
            prefix.append(column[0])
        assert prefix, 'Expected identical Popularity candidate metadata prefix'
        cache = make_prompt_cache(model)
        for start in range(0, len(prefix), 1024):
            model(mx.array([prefix[start:start + 1024]]), cache=cache)
            mx.eval([layer.state for layer in cache])
            mx.clear_cache()
        prefix_lengths[variant] = len(prefix)
        caches[variant] = cache
    return tokenized, prefix_lengths, caches


def generate_batch(model, tokenizer, tasks, tokenized, prefix_lengths, caches):
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler

    tokens = [tokenized[request['user_id'], variant] for request, variant in tasks]
    accepted = [i for i, prompt in enumerate(tokens) if len(prompt) + MAX_TOKENS <= CONTEXT_LIMIT]
    raw = {}
    started = time.perf_counter()
    if accepted:
        result = batch_generate(
            model, tokenizer, [tokens[i][prefix_lengths[tasks[i][1]]:] for i in accepted],
            prompt_caches=[caches[tasks[i][1]] for i in accepted],
            max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0))
        assert len(result.texts) == len(accepted)
        raw = dict(zip(accepted, result.texts))
    elapsed = time.perf_counter() - started
    return tokens, raw, elapsed


def infer(args):
    work_dir = args.output_dir
    requests = read_jsonl(work_dir / 'prompts.jsonl')
    protocol = json.loads((work_dir / 'expected_ranker_protocol.json').read_text())
    assert [request['user_id'] for request in requests] == protocol['users']
    requests = requests[:args.pilot_users] if args.pilot_users else requests
    out = work_dir / ('pilot' if args.pilot_users else 'inference')
    out.mkdir(exist_ok=True)
    parts = out / 'batches'
    parts.mkdir(exist_ok=True)
    model_files = {name: file_hash(args.model_dir / name) for name in (
        'config.json', 'tokenizer.json', 'tokenizer_config.json', *WEIGHT_SHA256)}
    for name, expected in WEIGHT_SHA256.items():
        if model_files[name] != expected:
            raise ValueError(f'Model shard {name} does not match the pinned revision')
    settings = {
        'model_id': MODEL, 'revision': REVISION,
        'model_files_sha256': model_files,
        'device': 'Apple GPU (MLX Metal)', 'hardware': args.hardware,
        'system': platform.platform(), 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers')},
        'dtype': 'bfloat16 (unquantized upstream weights)',
        'sampler_temperature': 0.0, 'thinking': False,
        'max_new_tokens': MAX_TOKENS, 'context_limit': CONTEXT_LIMIT,
        'prefill_cache': 'common token prefix, independently for each prompt variant',
        'eos_token_ids': [151645, 151643], 'history_events': HISTORY_EVENTS,
        'users_per_batch': args.users_per_batch,
        'pilot_users': args.pilot_users, 'protocol_hash': protocol['protocol_hash'],
        'prompts_sha256': file_hash(work_dir / 'prompts.jsonl'),
        'category_instruction_sha256': file_hash(ROOT / 'category_instruction.txt'),
    }
    metadata_path = out / 'settings.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != settings:
        raise ValueError(f'Existing batch records in {out} have different settings')
    write_json(metadata_path, settings)
    started = time.perf_counter()
    model = tokenizer = None
    by_variant = {variant: [] for variant in VARIANTS}
    cached_batches = 0
    generation_seconds = 0.0
    prefix_seconds = 0.0
    prefix_lengths = {}
    for start in range(0, len(requests), args.users_per_batch):
        chunk = requests[start:start + args.users_per_batch]
        batch = [(request, variant) for request in chunk for variant in VARIANTS]
        target = parts / f'{start:03d}.json'
        if target.exists():
            records = json.loads(target.read_text())
            assert [(r['user_id'], r['variant']) for r in records] == [
                (q['user_id'], v) for q, v in batch]
            cached_batches += 1
        else:
            if model is None:
                model, tokenizer = open_model(args.model_dir)
                tokenizer.eos_token_ids = settings['eos_token_ids']
                prefix_started = time.perf_counter()
                tokenized, prefix_lengths, caches = prepare_model_inputs(model, tokenizer, requests)
                prefix_seconds = time.perf_counter() - prefix_started
                print(f'{out.name}: cached {prefix_lengths} prefix tokens in {prefix_seconds:.2f}s',
                      flush=True)
            input_tokens, raw, elapsed = generate_batch(
                model, tokenizer, batch, tokenized, prefix_lengths, caches)
            records = []
            for i, (request, variant) in enumerate(batch):
                ids = request['candidate_ids']
                text = raw.get(i)
                repair = False
                error = None
                try:
                    if text is None:
                        raise ValueError('Prompt exceeds the context budget')
                    ranking, repair = ranker.parse_ranking(text, set(ids))
                except ValueError as exc:
                    ranking = ids[:10]
                    error = str(exc)
                records.append({
                    'user_id': request['user_id'], 'variant': variant,
                    'protocol_hash': protocol['protocol_hash'], 'stage': 'ranker',
                    'ranked_ids': ranking, 'raw_response': text,
                    'parse_repaired': repair, 'fallback': error is not None, 'error': error,
                    'input_tokens': len(input_tokens[i]), 'batch_start': start,
                    'batch_seconds': elapsed,
                })
            write_json(target, records)
        generation_seconds += records[0]['batch_seconds']
        for record in records:
            by_variant[record['variant']].append(record)
        print(f'{out.name}: {start + len(chunk)}/{len(requests)} users; '
              f'batch has {sum(r["fallback"] for r in records)} fallback replies', flush=True)

    for variant, records in by_variant.items():
        write_jsonl(out / f'{variant}_responses.jsonl', records)
    write_json(out / 'timing.json', {
        'inference_command_seconds': time.perf_counter() - started,
        'joint_generation_seconds': generation_seconds,
        'prefix_precompute_seconds': prefix_seconds,
        'prefix_tokens': prefix_lengths,
        'resumed_batches': cached_batches, 'users_per_variant': len(requests),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('prepare')
    preparation.add_argument('--data-dir', type=Path, required=True)
    preparation.add_argument('--baseline-dir', type=Path, required=True)
    preparation.add_argument('--output-dir', type=Path, required=True)
    inference = sub.add_parser('infer')
    inference.add_argument('--output-dir', type=Path, required=True)
    inference.add_argument('--model-dir', type=Path, required=True)
    inference.add_argument('--hardware', required=True)
    inference.add_argument('--users-per-batch', type=int, default=4)
    inference.add_argument('--pilot-users', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.users_per_batch < 1 or not 0 <= args.pilot_users < 100:
            parser.error('Expected positive batch size and pilot size from 0 to 99')
        infer(args)


if __name__ == '__main__':
    main()
