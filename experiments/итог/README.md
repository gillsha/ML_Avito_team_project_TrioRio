# A08 risk-aware reranker (TrioRio)

This adds only the paired Original/Risk prompt variant; it does not alter the shared starter, classical methods, Category Ranker, candidate selection, or evaluator. `rank_risk.py` calls the same `ranker.build_prompt()` used by `experiments/a08_gill/rank_a08.py`, then replaces profile values with source-verified values and serializes one shared payload for both instructions. The Risk condition appends exactly the text in `risk_instruction.txt`; `customerType` remains an ordinary shared payload field, not an instruction or a proxy for `riskLevel`.

Risk availability is established from the corresponding raw `customer_information.csv` record only when its documented `timestamp` (last update) is strictly before 2021-06-01. The latest such record is used. The placeholder timestamp `2000-01-01`, absent/missing values, `Not_Available`, and every `Predicted_*` value are masked to `unknown`; later snapshots are excluded. `lastQuestionnaireDate` is not treated as an availability timestamp. Users remain in the fixed 100-user cohort regardless.

## Setup and pilot

From repository root, with the same FAR-Trans extraction available locally:

```bash
export DATA="$HOME/path/to/FAR-Trans"
export EXP="$PWD/experiments/a08_katya"
python3.11 -m venv "$EXP/.venv"
source "$EXP/.venv/bin/activate"
python -m pip install -r experiments/a08_gill/requirements-mlx-lock.txt
python experiments/a08_gill/course/far-trans/starter.py run --data-dir "$DATA" --output-dir "$EXP/runs/baseline"
python "$EXP/rank_risk.py" prepare --data-dir "$DATA" --baseline-dir "$EXP/runs/baseline" --output-dir "$EXP/runs/paired"
HF_XET_HIGH_PERFORMANCE=1 python experiments/a08_gill/fetch_model.py "$EXP/model"
python "$EXP/rank_risk.py" infer --output-dir "$EXP/runs/paired" --model-dir "$EXP/model" --hardware 'Apple Silicon; describe exact chip/RAM' --pilot-users 3
```

For a deliberately balanced pilot chosen from the frozen cohort, pass `--user-ids ID1 ID2 ID3` instead of `--pilot-users`; the IDs must already belong to that cohort. Selection should be made from `profile_provenance.jsonl` only, without inspecting future labels. Supplying a partial `--user-ids` list always writes to `pilot/`, even if `--pilot-users 0` is also passed. Only the exact ordered 100-user cohort writes to `inference/`.

`--pilot-users 3` calls the real local open-weight model on three fixed cohort members for both conditions. `--pilot-users 0` with no `--user-ids` selects all 100; do not use it until the pilot is reviewed. After approval, the full run (and safe resume from completed answer journals) is:

```bash
python "$EXP/rank_risk.py" infer --output-dir "$EXP/runs/paired" \
  --model-dir "$EXP/model" --hardware 'Apple Silicon Apple M2; RAM bytes=17179869184' \
  --pilot-users 0
```

Each completed answer is atomically saved under `pilot/journal/` or `inference/journal/` and reused only when prompt, protocol, candidates and model-settings hashes match. Final variant JSONL files are atomically published only after every selected answer exists; full publication verifies the exact frozen order and all 100 unique users. GPU/model/generation exceptions are logged separately and stop inference, never turning into fallback. Invalid model text alone falls back to the first ten fixed Popularity candidates, retaining its raw text and reason. `artifacts_sha256.json` contains full SHA-256 digests. Four classical baselines are preserved in `runs/baseline`.

After the pilot is reviewed and a full inference is explicitly approved, replay each 100-user response file with the unchanged course evaluator (do not use the 3-user pilot file):

```bash
python experiments/a08_gill/course/far-trans/starter.py run --data-dir "$DATA" \
  --output-dir "$EXP/runs/original" --candidate-count 50 \
  --ranker-candidate-source popularity --responses "$EXP/runs/paired/inference/original_responses.jsonl" --on-invalid error
python experiments/a08_gill/course/far-trans/starter.py run --data-dir "$DATA" \
  --output-dir "$EXP/runs/risk" --candidate-count 50 \
  --ranker-candidate-source popularity --responses "$EXP/runs/paired/inference/risk_responses.jsonl" --on-invalid error
python "$EXP/rank_risk.py" analyze --paired-dir "$EXP/runs/paired" \
  --baseline-dir "$EXP/runs/baseline" --original-dir "$EXP/runs/original" \
  --risk-dir "$EXP/runs/risk" --output "$EXP/runs/subgroup_summary.json"
```

Both response files carry the starter's expected `stage`, `protocol_hash`, and `ranked_ids`; both replay protocols are the same as Elizaveta's ranker protocol hash. The replay preserves all four classical baselines and the starter's original nDCG@10 formula.

`subgroup_summary.json` reports both all-user and subgroup metrics, the paired Risk−Original and Risk−Popularity deltas, and separate valid-only diagnostics. The primary averages still include every fallback user at the fallback ranking; valid-only numbers are diagnostic and must not replace them.

Tests (Python 3.11 venv with the pinned MLX stack; the prefix-cache equivalence test uses a tiny random MLX model, not the Qwen weights):

```bash
python -m unittest discover -s experiments/a08_katya/tests -v
```

The final report must compare the baseline's four methods with Original and Risk on the same sample, show both all-user and nonempty nDCG@10, and calculate the stated risk/Professional subgroup only from pre-cutoff observed fields. Do not claim the hypothesis was tested if profile availability is unverified or the subgroup has inadequate observations.
