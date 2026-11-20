import json
import sys
import csv
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rank_risk


class RiskRankerTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'ISIN{i:03}' for i in range(50)]
        self.request = {'candidate_ids': self.ids}

    def test_original_and_risk_share_identical_candidates_and_payload_bytes(self):
        payload = {'candidate_assets':[{'id':i} for i in self.ids], 'profile_as_of_cutoff': {'riskLevel':'unknown'}}
        original, risk = rank_risk.paired_prompts('original instruction', payload, 'one added instruction')
        self.request.update(original=original, risk=risk)
        original = self.request['original'].split('\n', 1)[1]
        risk = self.request['risk'].split('\n', 1)[1]
        self.assertEqual(original.encode(), risk.encode())
        self.assertEqual([x['id'] for x in json.loads(original)['candidate_assets']], self.ids)

    def test_unknown_predicted_placeholder_and_late_profiles_are_masked(self):
        row = {'timestamp':'2021-05-01','riskLevel':'Predicted_Aggressive',
               'customerType':'Professional','investmentCapacity':'Not_Available'}
        safe = rank_risk.safe_profile(row)
        self.assertEqual(safe['riskLevel'], {'value':'unknown','status':'predicted'})
        self.assertEqual(safe['customerType']['value'], 'Professional')
        self.assertEqual(safe['investmentCapacity']['value'], 'unknown')
        for timestamp in ('2000-01-01', '2021-06-01', '2022-01-01'):
            self.assertEqual(rank_risk.safe_profile(dict(row, timestamp=timestamp))['riskLevel']['value'], 'unknown')
        self.assertEqual(rank_risk.safe_profile(None)['riskLevel']['status'], 'unverified')

    def test_later_snapshot_does_not_replace_pre_cutoff_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / 'customer_information.csv').write_text(
                'customerID,customerType,riskLevel,investmentCapacity,lastQuestionnaireDate,timestamp\n'
                'u1,Professional,Aggressive,CAP_GT300K,2020-02-01,2021-05-20\n'
                'u1,Mass,Balanced,CAP_LT_30K,2019-01-01,2021-06-15\n')
            profiles, late = rank_risk.latest_profiles(data, {'u1'})
            self.assertEqual(profiles['u1']['row']['timestamp'], '2021-05-20')
            self.assertEqual(rank_risk.safe_profile(profiles['u1']['row'])['customerType']['value'], 'Professional')
            self.assertEqual(len(late), 1)

    def test_ranking_requires_exactly_ten_unique_candidates(self):
        ranking, repaired, fallback, error = rank_risk.parse_or_fallback(
            self.request, json.dumps(self.ids[:10]))
        self.assertEqual(ranking, self.ids[:10])
        self.assertFalse(fallback)
        self.assertEqual(len(set(ranking)), 10)

    def test_invalid_response_falls_back_but_keeps_user_and_raw_error(self):
        cases = [
            ('not json', 'No JSON array found'),
            (json.dumps(self.ids[:9]), 'exactly ten'),
            (json.dumps(self.ids[:9] + [self.ids[0]]), 'Duplicate'),
            (json.dumps(self.ids[:9] + ['FOREIGN']), 'outside candidate'),
            (json.dumps([self.ids[0]] * 10), 'Duplicate'),
        ]
        for raw, reason in cases:
            with self.subTest(raw=raw):
                ranked, _, fallback, error = rank_risk.parse_or_fallback(self.request, raw)
                self.assertEqual(ranked, self.ids[:10])
                self.assertTrue(fallback)
                self.assertIn(reason, error)
                # Caller persists both exact raw text and the reason without dropping user.
                response = {'user_id':'kept', 'raw_response':raw, 'fallback':fallback, 'error':error}
                self.assertEqual(response['user_id'], 'kept')
                self.assertEqual(response['raw_response'], raw)

    def test_future_transactions_are_excluded_by_strict_cutoff_predicate(self):
        history = [('2021-05-31','x','Buy'), ('2021-06-01','y','Buy'), ('2021-06-02','z','Buy')]
        admitted = [event for event in history if event[0][:10] < rank_risk.CUTOFF]
        self.assertEqual(admitted, [history[0]])

    def test_course_data_loader_excludes_transactions_at_and_after_cutoff(self):
        starter_path = rank_risk.GILL / 'course' / 'far-trans' / 'starter.py'
        import importlib.util
        spec = importlib.util.spec_from_file_location('test_starter_cutoff', starter_path)
        starter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(starter)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {
                'asset_information.csv': ('ISIN,assetName,assetCategory,assetSubCategory,sector,industry,marketID,timestamp\n'
                                          'A,Asset,Stock,Common,Sector,Industry,M,2020-01-01\n'),
                'customer_information.csv': ('customerID,customerType,riskLevel,investmentCapacity,lastQuestionnaireDate,timestamp\n'
                                             'U,Mass,Balanced,CAP_30K_80K,2020-01-01,2020-01-01\n'),
                'close_prices.csv': 'ISIN,timestamp,closePrice\nA,2020-01-01,1\n',
                'transactions.csv': ('customerID,ISIN,transactionID,transactionType,timestamp,totalValue,units,channel,marketID\n'
                                     'U,A,1,Buy,2021-05-31,1,1,Internet Banking,M\n'
                                     'U,A,2,Buy,2021-06-01,1,1,Internet Banking,M\n'
                                     'U,A,3,Buy,2021-06-02,1,1,Internet Banking,M\n'),
            }
            for name, text in files.items():
                (root / name).write_text(text)
            _, _, history, truth, _, _ = starter.load_data(root, '2021-06-01', 30)
            self.assertEqual(history['U']['A'], 1)
            self.assertEqual(truth['U'], {'A'})

    def test_analysis_preserves_classic_methods_and_separates_custom_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {'users':['u1','u2'], 'cutoff':'2021-06-01', 'eligible_assets':['A']}
            ranker_protocol = dict(base, ranker_candidate_source='popularity',
                                   ranker_candidates_sha256='candidates', protocol_hash='ranker-hash')
            for name, protocol in (('baseline', dict(base, protocol_hash='base-hash')),
                                   ('original', ranker_protocol), ('risk', ranker_protocol)):
                folder = root / name
                folder.mkdir()
                (folder / 'protocol.json').write_text(json.dumps(protocol))
                methods = ['popularity','cbf','latest_buy','latest_buy_or_sell']
                if name != 'baseline':
                    methods.append('custom')
                with (folder / 'per_user.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['user_id','method','relevant_count','ndcg_at_10'])
                    writer.writeheader()
                    for method in methods:
                        writer.writerow({'user_id':'u1','method':method,'relevant_count':1,'ndcg_at_10':1})
                        writer.writerow({'user_id':'u2','method':method,'relevant_count':0,'ndcg_at_10':0})
            paired = root / 'paired'
            paired.mkdir()
            (paired / 'protocol.json').write_text(json.dumps(ranker_protocol))
            (paired / 'profile_provenance.jsonl').write_text('\n'.join([
                json.dumps({'user_id':'u1','safe_profile':{'riskLevel':{'status':'observed','value':'Aggressive'},'customerType':{'status':'observed','value':'Mass'}}}),
                json.dumps({'user_id':'u2','safe_profile':{'riskLevel':{'status':'observed','value':'Balanced'},'customerType':{'status':'observed','value':'Professional'}}})
            ])+'\n')
            inference = paired / 'inference'
            inference.mkdir()
            for variant in ('original', 'risk'):
                (inference / f'{variant}_responses.jsonl').write_text('\n'.join(
                    json.dumps({'user_id':uid, 'fallback':False}) for uid in ('u1','u2'))+'\n')
            output = root / 'summary.json'
            with contextlib.redirect_stdout(io.StringIO()):
                rank_risk.analyze(argparse.Namespace(paired_dir=paired,
                    baseline_dir=root/'baseline', original_dir=root/'original', risk_dir=root/'risk', output=output))
            summary = json.loads(output.read_text())
            self.assertEqual(summary['subgroup_users'], 2)
            self.assertEqual(summary['methods']['original_custom']['original_risk_or_professional']['users'], 2)
            self.assertEqual(summary['methods']['risk_custom']['risk_all']['ndcg_at_10'], .5)
            self.assertIn('latest_buy_or_sell', summary['methods'])
            self.assertEqual(summary['comparisons']['risk_or_professional']['risk_minus_original_ndcg_at_10'], 0)
            self.assertEqual(summary['fallback_counts_kept_in_primary_metrics'], {'original':0, 'risk':0})

    def _persist_valid_answer(self, directory, request, variant='original', settings_hash='settings-A'):
        prompt = request[variant]
        expected = {'user_id':request['user_id'], 'variant':variant, 'protocol_hash':'protocol-A',
                    'prompt_sha256':rank_risk.hashlib.sha256(prompt.encode()).hexdigest(),
                    'settings_hash':settings_hash, 'candidate_ids':request['candidate_ids']}
        record = dict(expected, stage='ranker', prompt=prompt, raw_response=json.dumps(request['candidate_ids'][:10]),
                      ranked_ids=request['candidate_ids'][:10], fallback=False, error=None, parse_repaired=False)
        path = rank_risk.journal_path(directory, request['user_id'], variant)
        rank_risk.atomic_json(path, record)
        return path, expected, record

    def test_atomic_intermediate_save_resume_and_ordered_full_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'inference'
            cohort = ['u1','u2']
            selected = [{'user_id':uid, 'candidate_ids':self.ids, 'original':f'prompt {uid}', 'risk':f'risk {uid}'} for uid in cohort]
            path, expected, saved = self._persist_valid_answer(run, selected[0])
            self._persist_valid_answer(run, selected[0], variant='risk')
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name+'.tmp').exists())
            self.assertEqual(rank_risk.compatible_answer(path, expected, self.ids), saved)
            with self.assertRaisesRegex(ValueError, 'Incompatible saved answer'):
                rank_risk.compatible_answer(path, dict(expected, settings_hash='settings-B'), self.ids)
            # Simulate restart: completed u1 is reused, missing u2 prevents publication.
            with self.assertRaisesRegex(ValueError, 'missing'):
                rank_risk.publish_results(run, 'original', selected, 'protocol-A', 'settings-A', cohort, require_full=True)
            self.assertFalse((run/'original_responses.jsonl').exists())
            self._persist_valid_answer(run, selected[1])
            self._persist_valid_answer(run, selected[1], variant='risk')
            for variant in ('original', 'risk'):
                records = rank_risk.publish_results(run, variant, selected, 'protocol-A', 'settings-A', cohort,
                                                    require_full=True)
                self.assertEqual([r['user_id'] for r in records], cohort)
                lines = (run/f'{variant}_responses.jsonl').read_text().splitlines()
                self.assertEqual([json.loads(line)['user_id'] for line in lines], cohort)
                self.assertEqual(len(lines), len(cohort))

    def test_invalid_model_text_is_atomically_saved_with_fallback_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            request = {'user_id':'u1', 'candidate_ids':self.ids,
                       'original':'original instruction\n{}', 'risk':'risk instruction\n{}'}
            path, expected, _ = self._persist_valid_answer(run, request)
            record = json.loads(path.read_text())
            raw = 'not JSON from the model'
            ranking, repaired, fallback, error = rank_risk.parse_or_fallback(request, raw)
            record.update(raw_response=raw, ranked_ids=ranking, fallback=fallback,
                          error=error, parse_repaired=repaired)
            rank_risk.atomic_json(path, record)
            restored = rank_risk.compatible_answer(path, expected, self.ids)
            self.assertEqual(restored['raw_response'], raw)
            self.assertTrue(restored['fallback'])
            self.assertEqual(restored['ranked_ids'], self.ids[:10])
            self.assertEqual(restored['error'], 'No JSON array found')

    def test_partial_user_ids_never_map_to_full_even_when_pilot_users_zero(self):
        full, full_path = rank_risk.run_location(Path('/exp'), ['u1','u2'], ['u1','u2'])
        partial, partial_path = rank_risk.run_location(Path('/exp'), ['u2'], ['u1','u2'])
        self.assertTrue(full)
        self.assertEqual(full_path, Path('/exp/inference'))
        self.assertFalse(partial)
        self.assertEqual(partial_path, Path('/exp/pilot'))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'inference'
            selected = [{'user_id':'u2', 'candidate_ids':self.ids, 'original':'prompt', 'risk':'risk'}]
            with self.assertRaisesRegex(ValueError, 'exact frozen user order'):
                rank_risk.publish_results(run, 'original', selected, 'protocol-A', 'settings-A', ['u1','u2'],
                                          require_full=True)
            self.assertFalse((run/'original_responses.jsonl').exists())

    def test_generation_exception_is_infrastructure_error_not_fallback(self):
        def fail():
            raise MemoryError('Metal out of memory')
        with self.assertRaisesRegex(rank_risk.InferenceInfrastructureError, 'Metal out of memory'):
            rank_risk.call_generator(fail)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            exc = rank_risk.InferenceInfrastructureError('GPU failure')
            rank_risk.save_infrastructure_error(run, 'u1', 'risk', 'generation', exc)
            error = json.loads(next((run/'infrastructure_errors').rglob('*.json')).read_text())
            self.assertFalse(error['fallback'])
            self.assertNotIn('ranked_ids', error)

    def test_common_prefix_cached_logits_match_full_prompt_toy_mlx(self):
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models import llama
        from mlx_lm.models.cache import make_prompt_cache
        class TinyTokenizer:
            eos_token_ids = [63]
            @staticmethod
            def decode(tokens):
                return ','.join(map(str, tokens))
        mx.random.seed(23)
        args = llama.ModelArgs(model_type='llama', hidden_size=16, num_hidden_layers=2,
            intermediate_size=32, num_attention_heads=2, num_key_value_heads=2,
            vocab_size=64, rms_norm_eps=1e-5, max_position_embeddings=128)
        model = llama.Model(args)
        mx.eval(model.parameters())
        prompts = [[1,2,3,4,5,6], [1,2,3,4,7,8]]
        prefix = rank_risk.common_token_prefix(prompts)
        self.assertEqual(prefix, [1,2,3,4])
        full_cache = make_prompt_cache(model)
        full_logits = model(mx.array([prompts[0]]), cache=full_cache)
        prefix_cache = make_prompt_cache(model)
        model(mx.array([prefix]), cache=prefix_cache)
        mx.eval([layer.state for layer in prefix_cache])
        cached_logits = model(mx.array([prompts[0][len(prefix):]]), cache=prefix_cache)
        mx.eval(full_logits, cached_logits)
        delta = mx.max(mx.abs(full_logits[:,-1,:] - cached_logits[:,-1,:])).item()
        self.assertLess(float(delta), 1e-6)
        self.assertEqual(int(mx.argmax(full_logits[:,-1,:]).item()),
                         int(mx.argmax(cached_logits[:,-1,:]).item()))
        uncached = [batch_generate(model, TinyTokenizer(), [prompt], max_tokens=4,
                                   sampler=make_sampler(temp=0.0)).texts[0] for prompt in prompts]
        fresh_cache = make_prompt_cache(model)
        model(mx.array([prefix]), cache=fresh_cache)
        mx.eval([layer.state for layer in fresh_cache])
        cached = [batch_generate(model, TinyTokenizer(), [prompt[len(prefix):]],
            prompt_caches=[fresh_cache], max_tokens=4,
            sampler=make_sampler(temp=0.0)).texts[0] for prompt in prompts]
        self.assertEqual(uncached, cached)


if __name__ == '__main__':
    unittest.main()
