# Boolean Query Formulation

This is an LLM-based, concept-based formulation adaptation for the MAESTRO
retrieval pipeline.  It is **not** a reproduction of a published algorithm.

## What it does

The existing `ReformulationAgent` (action `QueryReform`, id 0) can now run in
two modes:

* `natural` (default): one LLM call rewrites the query as natural language.
* `boolean`: one LLM call extracts required concepts and renders them as an
  explicit Boolean query, e.g.

```text
("electric vehicle" OR "electric car") AND ("battery" OR "batteries") AND ("recycling")
```

The Boolean agent is query-only: it uses **only** the original query text for
formulation.  It does not inspect retrieved documents.  This keeps formulation
semantics clean and lets the downstream policy still observe the effect of the
reformulation through the normal state-building retrieval.

## Files changed / added

* `src/agents/boolean_formulate.py` — new agent.
* `src/agents/reformulate.py` — `formulation_mode` config, delegates to Boolean
  agent in `boolean` mode.
* `src/agents/__init__.py` — exports `BooleanFormulationAgent`.
* `src/simulation.py` — propagates `new_corpus_data` so that subsequent actions
  (rerank, PRF) see document text from the ranking produced by the current
  action.
* `src/agents/prf.py` — returns `new_corpus_data` for its re-retrieved ranking.
* `src/main.py` — `--qr-formulation {natural,boolean}` CLI option; run name
  includes the mode so checkpoints do not collide.
* `generate_baseline_traces.py` — `--qr-formulation` option for the
  `reformulate` baseline.
* `evaluate_formulation.py` — lightweight query-first evaluation of original,
  natural, and Boolean baselines.
* `tests/test_boolean_formulation.py` — mocked unit tests.
* `docs/boolean_formulation.md` — this file.

## Backend syntax support

ChatNoir's simple search endpoint (`/api/v1/_search`) documents support for the
standard web-search operators `AND`, `OR`, `-`, `"..."` phrases, `site:...`,
etc.  The Boolean renderer therefore uses:

* `OR` between alternatives inside a concept group,
* `AND` between required concept groups,
* double quotes around multi-word alternatives,
* parentheses around each group to make precedence explicit.

**Important caveat:** the public documentation confirms `AND`, `OR`, phrases
and the minus operator, but it does not explicitly confirm parenthesised
grouping.  Parentheses are standard search-engine syntax and are used here to
avoid precedence ambiguity when mixing `AND` and `OR`, but we have not
experimentally verified that ChatNoir parses grouped expressions exactly as
intended.  If you run the live evaluation and see Boolean queries returning
unexpected result counts or rankings, this is the first thing to check.

The default `search_method` is plain `bm25`; the operators above are supported
by the simple search module, not only by the default retrieval model.

## Configuration

### In `src/main.py`

```bash
poetry run python src/main.py \
  --experiment-dir experiments/my_run \
  --policy random \
  --qr-formulation boolean
```

### In `generate_baseline_traces.py`

```bash
poetry run python generate_baseline_traces.py --qr-formulation boolean
```

### In `evaluate_formulation.py`

```bash
poetry run python evaluate_formulation.py --max-queries 10
```

The evaluation script compares three baselines on the same query IDs used by
the CQL traces:

1. `original` — retrieve the raw query once.
2. `natural` — one LLM rewrite + one retrieval.
3. `boolean` — one concept extraction + one retrieval.

Formulation baselines do **not** retrieve the original query first.  The CSV
output logs the generated query, formulation mode, fallback/failure reason,
retrieval-call count, number of results, NDCG@k, Recall@k and latency.

## Run tests

Tests are mocked and do not require a live API key or LLM endpoint:

```bash
poetry run python -m unittest tests.test_boolean_formulation -v
```

Live backend tests are intentionally opt-in (run `evaluate_formulation.py`
with a valid `CHATNOIR_API_KEY`).

## Design notes

* **Policy compatibility preserved.**  Action IDs, action count, state
  dimensions and default behaviour remain unchanged.  Only the internal
  behaviour of the reformulation action is configurable.
* **No retraining required.**  The mode is selected at trajectory-generation
  or evaluation time.
* **Fallback is explicit and logged.**  If the LLM returns malformed JSON, the
  structure is invalid, or rendering fails, the agent falls back to the
  original query and records the reason.  It does not silently relax the
  Boolean expression.
* **NOT / exclusions are not implemented.**  The first version only builds
  conjunctive required-concept groups with internal alternatives.
* **Corpus propagation fixed.**  Reformulation and PRF now return the document
  text mapping for their new rankings, and `Simulation` forwards it to the
  next action.  Previously the corpus from the initial retrieval was reused
  after reformulation.

## Limitations

* Boolean operator precedence relies on ChatNoir parsing grouped expressions
  as expected; this has not been experimentally verified.
* The LLM may occasionally miss a concept or produce overly broad
  alternatives.  Validation rejects empty/malformed output and excessive
  expansion, but it cannot guarantee semantic correctness.
* Boolean queries are typically more restrictive than natural-language
  queries; zero-result valid queries are possible and are logged separately
  from retrieval errors.
* Changing formulation behaviour changes action semantics.  Datasets and
  policy checkpoints generated with `natural` reformulation are **not**
  automatically equivalent to those generated with `boolean` reformulation.
  Use a separate experiment directory / run name when switching modes.
* The implementation does not add NOT/exclusions, fielded search, or term
  weighting.
