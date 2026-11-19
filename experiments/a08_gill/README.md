# FAR-Trans Homework 2 · Elizaveta Gill

The changed prompt explicitly asks the Ranker to use prior purchase categories. The original-instruction control uses the same model, payload, ordered candidates and generation settings. Both use the first 50 assets from the course Popularity implementation. The official four classical methods keep their own rankings.

## Sources and changes

- [Official course starter and Ranker](https://github.com/dryjins/RecSys-LLMs/tree/f6850861ea20a51e1fb018ee4e5f840c2ac759e0/project), commit `f6850861ea20a51e1fb018ee4e5f840c2ac759e0`. The selected original sources and MIT license are in `course/`. The only edit to course files is documented in `starter-popularity.patch` and adds an opt-in Popularity candidate source for ranker response replay. Default classical runs are unchanged.
- [Official FAR-Trans archive and license](https://researchdata.gla.ac.uk/1658/). The measured archive SHA-256 is `d0d6cfe027732bab45fdcf9af80eff4fefddc789aedae9bdce0d3d6cfd05cfb6`.
- [Exact Qwen3-1.7B snapshot](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e). Use native BF16 weights. Inference verifies both published shard hashes. No external inference endpoint is called.
- `category_instruction.txt` contains the entire prompt edit. `rank_a08.py` calls the upstream original prompt builder and appends the instruction immediately before the same JSON in both conditions. The adapter places the shared candidate records first and serializes the JSON compactly. MLX computes their common token prefix once per condition and reuses its key/value cache without dropping candidates or user fields. The 15 most recent dated Buy events and counts by ISIN for all eligible past buys retain the original prompt layout. Missing categories use the upstream `unknown` token and supply no match evidence. Without usable purchase evidence, the instruction asks to retain prior order for indistinguishable assets. The warm sample requires purchase history.

## Reproduce

Python 3.11 on Apple Silicon with Metal is the measured setup. The starter and response replay require the standard library only. Run these commands from this directory. Use a fresh results directory for an independent benchmark. A partial inference can resume from matching completed batch files.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlx-lock.txt
mkdir -p data
curl --fail --location https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip -o data/FAR-Trans.zip
unzip -q data/FAR-Trans.zip -d data
python fetch_model.py model
./run.sh "$PWD/data/FAR-Trans" "$PWD/model" "$PWD/runs/a08" 'Apple M3, 24 GB'
```

`run.sh` sets `PYTHONHASHSEED=0` for stable content-feature accumulation in the course implementation. It measures the default classical run, prepares and audits prompts, checks three users with real model calls, runs both variants across all 100 users, replays both response files through the starter, and recomputes summary metrics and per-asset ranks. The pipeline aborts on infrastructure errors and preserves completed batches. The upstream parser accepts exact JSON and can repair a JSON fence or an extractable array. A malformed response, wrong number or type of IDs, duplicated ID, ID outside the candidate list or overlong input uses the first 10 Popularity candidates. Every fallback stays in the cohort and its reason is recorded.

## Evidence files

The paired `a08_results` directory contains the actual measured outputs. With that directory and the official CSVs, the replay and audit can be run without model weights. To replay one variant, substitute `original` for `category` in all paths if desired.

```bash
PYTHONHASHSEED=0 python course/far-trans/starter.py run --data-dir data/FAR-Trans \
  --output-dir runs/category-replayed --ranker-candidate-source popularity \
  --responses ../a08_results/inference/category_responses.jsonl --on-invalid error
```

- `baseline/`, `category/`, `original/`: the starter outputs, including source and protocol hashes, top-10 rankings and per-user evaluation.
- `expected_ranker_protocol.json` and `prompts.jsonl`: frozen candidate lists, full paired prompts, and the extended protocol. The two Ranker runs share a protocol hash. The classical protocol equals their common fields, with two additional Ranker-specific fields documented by the analyzer.
- `pilot/` and `inference/`: settings, timing, raw model text, normalized response IDs and fallback records.
- `input_audit.json`: raw-row timestamp verification and short-history counts. Short history means one or two pre-cutoff Buy/Sell transactions, across all assets, after exact-row deduplication and exclusion of negative transaction IDs.
- `summary.json`, `purchased_asset_ranks.csv`: full-cohort and subgroup results, timing, fallbacks, candidate recall and rank movements.
- `manual_verification_ru.md`: instructions for the student's independent check.

User-level outputs contain pseudonymous customer IDs and are kept in the private evidence package. The Git version contains experiment code and aggregate results.

## Measured results

| Method | Users | Future buyers | nDCG@10 all | nDCG@10 nonempty | Failures |
|---|---:|---:|---:|---:|---:|
| Popularity | 100 | 11 | 0.0286 | 0.2599 | 0 |
| CBF | 100 | 11 | 0.0201 | 0.1827 | 0 |
| Latest buy | 100 | 11 | 0.0277 | 0.2520 | 0 |
| Latest buy or sell | 100 | 11 | 0.0271 | 0.2464 | 0 |
| Category Ranker | 100 | 11 | 0.0225 | 0.2049 | 15 |
| Original-instruction Ranker | 100 | 11 | 0.0336 | 0.3057 | 16 |

[Aggregate results](aggregate_results.json) include the short-history diagnostic, wall timings, exact hashes, and generation settings.
