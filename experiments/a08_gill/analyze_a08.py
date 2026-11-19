"""Audit prompt time boundaries and summarize the completed starter evaluations."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rank_a08 import ROOT, digest, read_jsonl, write_json, ranker

CLASSICAL = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')
VARIANTS = ('category', 'original')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def load_protocol(path):
    protocol = json.loads(path.read_text())
    assert protocol['protocol_hash'] == digest({
        key: value for key, value in protocol.items() if key != 'protocol_hash'})
    return protocol


def sample_transactions(data_dir, protocol, example_user):
    cutoff = protocol['cutoff']
    end = protocol['end_exclusive']
    eligible, users = set(protocol['eligible_assets']), set(protocol['users'])
    counts, purchases = Counter(), defaultdict(Counter)
    recent, truth, examples = defaultdict(list), defaultdict(set), defaultdict(list)
    seen = set()
    for line, row in enumerate(rows(data_dir / 'transactions.csv'), 2):
        uid, isin = row['customerID'], row['ISIN']
        if uid not in users or int(row['transactionID']) < 0:
            continue
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        date = row['timestamp'][:10]
        kind = row['transactionType']
        if kind not in {'Buy', 'Sell'}:
            continue
        if date < cutoff:
            counts[uid] += 1
            if kind == 'Buy' and isin in eligible:
                purchases[uid][isin] += 1
                recent[uid].append((date, isin))
                if uid == example_user:
                    examples['past'].append(dict(line=line, date=date, ISIN=isin, event=kind))
        elif date < end and kind == 'Buy' and isin in eligible:
            truth[uid].add(isin)
            if uid == example_user:
                examples['future'].append(dict(line=line, date=date, ISIN=isin, event=kind))
    examples['past'] = sorted(examples['past'], key=lambda r: (r['date'], r['ISIN']))[-2:]
    return counts, purchases, recent, truth, dict(examples)


def audit(data_dir, results, example_user=None):
    protocol = load_protocol(results / 'expected_ranker_protocol.json')
    users, cutoff = protocol['users'], protocol['cutoff']
    if example_user is not None and example_user not in users:
        raise ValueError('Example user must belong to the frozen cohort')
    counts, purchases, recent, truth, examples = sample_transactions(data_dir, protocol, example_user)
    assets, profiles = {}, {}
    excluded_profile = None
    for file, key, snapshot in (
        ('asset_information.csv', 'ISIN', assets),
        ('customer_information.csv', 'customerID', profiles),
    ):
        for line, row in enumerate(rows(data_dir / file), 2):
            prior = snapshot.get(row[key])
            if row['timestamp'][:10] < cutoff and (prior is None or prior[1]['timestamp'] < row['timestamp']):
                snapshot[row[key]] = (line, row)
            elif (file == 'customer_information.csv' and row[key] in users
                  and row['timestamp'][:10] >= cutoff and excluded_profile is None):
                excluded_profile = {'user_id': row[key], 'line': line, 'date': row['timestamp']}
    if excluded_profile:
        line, past = profiles[excluded_profile['user_id']]
        examples['profile_time_check'] = {
            'older_source_line_used': line, 'older_date_used': past['timestamp'],
            'later_row_excluded': excluded_profile,
        }

    requests = read_jsonl(results / 'prompts.jsonl')
    assert [r['user_id'] for r in requests] == users
    assert protocol['ranker_candidates_sha256'] == digest({
        record['user_id']: record['candidate_ids'] for record in requests})
    checked_fields = 0
    history_categories = Counter()
    common_candidates = json.loads(requests[0]['original'].split('\n', 1)[1])['candidate_assets']
    for request in requests:
        uid = request['user_id']
        text = request['original'].split('\n', 1)[1]
        assert request['category'].split('\n', 1)[1] == text
        payload = json.loads(text)
        assert set(payload) == {
            'buy_history_counts', 'candidate_assets', 'chronology_note', 'cutoff_exclusive',
            'most_recent_buy', 'profile_as_of_cutoff',
            'recent_buy_events_oldest_to_newest', 'repeat_purchases_allowed'}
        assert payload['cutoff_exclusive'] == cutoff and payload['repeat_purchases_allowed']
        assert [a['ISIN'] for a in payload['candidate_assets']] == request['candidate_ids']
        assert payload['candidate_assets'] == common_candidates
        assert payload['profile_as_of_cutoff']['timestamp'] == profiles[uid][1]['timestamp']
        for key in ('customerType', 'riskLevel', 'investmentCapacity'):
            assert payload['profile_as_of_cutoff'][key]['value'] == profiles[uid][1][key]
        assert Counter({a['ISIN']: a['buy_count'] for a in payload['buy_history_counts']}) == purchases[uid]
        known_categories = {a['assetCategory'] for a in payload['buy_history_counts']
                            if a['assetCategory'] != 'unknown'}
        history_categories[len(known_categories)] += 1
        assert [(r['date'], r['ISIN']) for r in payload['recent_buy_events_oldest_to_newest']] == sorted(recent[uid])[-15:]
        assert payload['most_recent_buy'] == payload['recent_buy_events_oldest_to_newest'][-1]
        for record in payload['buy_history_counts'] + payload['candidate_assets']:
            line, source = assets[record['ISIN']]
            for key in ranker.ASSET_FIELDS:
                assert record[key] == ranker.known(source.get(key))
                checked_fields += 1
            if uid == example_user and record['ISIN'] == payload['most_recent_buy']['ISIN']:
                examples['asset_metadata'] = dict(line=line, ISIN=record['ISIN'],
                                                   date=record['timestamp'],
                                                   category=record['assetCategory'])
    subgroup = [uid for uid in users if 1 <= counts[uid] <= 2]
    output = {
        'users': len(users), 'users_with_nonempty_truth': sum(bool(truth[uid]) for uid in users),
        'short_history_users': len(subgroup), 'short_history_users_with_nonempty_truth': sum(
            bool(truth[uid]) for uid in subgroup),
        'short_history_counts': {uid: counts[uid] for uid in users},
        'asset_fields_compared_to_source': checked_fields,
        'candidate_categories': dict(Counter(a['assetCategory'] for a in common_candidates)),
        'users_by_distinct_known_buy_categories': dict(history_categories),
        'checks': 'Original and category payloads byte-identical; Buy history and latest pre-cutoff asset and profile fields verified against raw rows.',
        'example_user': example_user, 'example_rows': examples,
    }
    write_json(results / 'input_audit.json', output)
    print({key: value for key, value in output.items() if key not in {'short_history_counts', 'example_rows'}})
    return protocol, truth, subgroup


def starter_predictions(directory):
    predictions = defaultdict(lambda: defaultdict(list))
    for row in rows(directory / 'predictions.csv'):
        ranking = predictions[row['method']][row['user_id']]
        assert int(row['rank']) == len(ranking) + 1
        ranking.append(row['ISIN'])
    return dict(predictions)


def elapsed(log):
    return float(next(line[5:] for line in log.read_text().splitlines() if line.startswith('real ')))


def analyze(data_dir, results, example_user=None):
    protocol, truth, subgroup = audit(data_dir, results, example_user)
    users = protocol['users']
    baseline = load_protocol(results / 'baseline' / 'protocol.json')
    fields_added = {'ranker_candidate_source', 'ranker_candidates_sha256', 'protocol_hash'}
    assert {k: v for k, v in protocol.items() if k not in fields_added} == {
        k: v for k, v in baseline.items() if k != 'protocol_hash'}
    predictions = starter_predictions(results / 'baseline')
    assert set(predictions) == set(CLASSICAL)
    failures = {method: 0 for method in CLASSICAL}
    response_details, replay_seconds = {}, {}
    candidates = {r['user_id']: r['candidate_ids'] for r in read_jsonl(results / 'prompts.jsonl')}
    for variant in VARIANTS:
        directory = results / variant
        assert load_protocol(directory / 'protocol.json') == protocol
        replay = starter_predictions(directory)
        assert all(predictions[method] == replay[method] for method in CLASSICAL)
        predictions[variant] = replay['custom']
        records = read_jsonl(results / 'inference' / f'{variant}_responses.jsonl')
        assert [r['user_id'] for r in records] == users
        for record in records:
            uid = record['user_id']
            assert record['protocol_hash'] == protocol['protocol_hash']
            ids = candidates[uid]
            assert len(ids) == 50 and predictions['popularity'][uid] == ids[:10]
            top = predictions[variant][uid]
            assert top == record['ranked_ids'] and len(top) == len(set(top)) == 10
            assert set(top) <= set(ids)
            if record['fallback']:
                assert top == ids[:10] and record['error']
        failures[variant] = sum(r['fallback'] for r in records)
        response_details[variant] = {
            'raw_model_responses': sum(r['raw_response'] is not None for r in records),
            'valid': len(records) - failures[variant],
            'parse_repairs': sum(r['parse_repaired'] for r in records),
            'fallback_reasons': dict(Counter(r['error'] for r in records if r['fallback'])),
        }
        replay_seconds[variant] = elapsed(results / f'{variant}.log')

    def score(ranking, relevant):
        dcg = math.fsum(1 / math.log2(rank + 1) for rank, isin in enumerate(ranking, 1)
                        if rank <= 10 and isin in relevant)
        ideal = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        return dcg / ideal if ideal else 0

    def aggregate(group):
        methods = {}
        for name, per_user in predictions.items():
            assert set(users) == set(per_user)
            scores = [(uid, score(per_user[uid], truth[uid])) for uid in group]
            nonempty = [value for uid, value in scores if truth[uid]]
            methods[name] = {
                'users': len(group), 'nonempty_truth_users': len(nonempty),
                'ndcg_at_10': math.fsum(value for _, value in scores) / len(group),
                'ndcg_at_10_nonempty': math.fsum(nonempty) / len(nonempty) if nonempty else None}
        return methods

    all_users = aggregate(users)
    for plan in ('baseline', *VARIANTS):
        metrics = json.loads((results / plan / 'metrics.json').read_text())['methods']
        for name, metric in metrics.items():
            expected = all_users[plan if name == 'custom' else name]
            assert metric['users'] == expected['users']
            assert metric['nonempty_truth_users'] == expected['nonempty_truth_users']
            for field in ('ndcg_at_10', 'ndcg_at_10_nonempty'):
                assert math.isclose(metric[field], expected[field], abs_tol=1e-14)
        for row in rows(results / plan / 'per_user.csv'):
            uid = row['user_id']
            name = plan if row['method'] == 'custom' else row['method']
            assert int(row['relevant_count']) == len(truth[uid])
            assert math.isclose(float(row['ndcg_at_10']), score(predictions[name][uid], truth[uid]),
                                abs_tol=1e-14)

    def position(ids, isin):
        return ids.index(isin) + 1 if isin in ids else None

    movement = []
    changes = Counter()
    for uid in users:
        for isin in sorted(truth[uid]):
            candidate = position(candidates[uid], isin)
            pop_rank = position(predictions['popularity'][uid], isin)
            ranks = {name: position(predictions[name][uid], isin) for name in VARIANTS}
            changes['truth_pairs'] += 1
            changes['in_top50' if candidate else 'outside_top50'] += 1
            new_rank = ranks['category']
            if new_rank != pop_rank:
                changes['rank_improved' if (new_rank or 11) < (pop_rank or 11) else 'rank_worsened'] += 1
            if new_rank and not pop_rank:
                changes['entered_top10'] += 1
            if pop_rank and not new_rank:
                changes['exited_top10'] += 1
            movement.append({'user_id': uid, 'ISIN': isin, 'popularity_candidate_rank': candidate,
                             'popularity_top10_rank': pop_rank,
                             'category_top10_rank': new_rank,
                             'original_top10_rank': ranks['original'],
                             'short_history': uid in subgroup})
    for key in ('rank_improved', 'rank_worsened', 'entered_top10', 'exited_top10', 'in_top50', 'outside_top50'):
        changes.setdefault(key, 0)
    with (results / 'purchased_asset_ranks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(movement[0]))
        writer.writeheader()
        writer.writerows(movement)

    result = {'all': all_users, 'short': aggregate(subgroup), 'failures': failures,
              'response_details': response_details, 'movement_counts': dict(changes),
              'baseline_wall_seconds': elapsed(results / 'baseline.log'),
              'inference_wall_seconds': elapsed(results / 'inference.log'),
              'inference_joint_timing': json.loads((results / 'inference' / 'timing.json').read_text()),
              'replay_seconds': replay_seconds, 'eligible_assets': len(protocol['eligible_assets']),
              'baseline_protocol_hash': baseline['protocol_hash'],
              'ranker_protocol_hash': protocol['protocol_hash'],
              'candidate_hash': protocol['ranker_candidates_sha256']}
    write_json(results / 'summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'analyze'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--example-user', help='Optional cohort member for source-row evidence')
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args.data_dir, args.results, args.example_user)
    else:
        analyze(args.data_dir, args.results, args.example_user)


if __name__ == '__main__':
    main()
