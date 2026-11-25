# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Symptom-based disease prediction comparing three retrieval sources: a Neo4j knowledge graph (`kg`), a LlamaIndex vector store (`vector`) and a DuckDuckGo ReAct web agent (`web`). A final Groq LLM call merges their answers into one report. The code is an installable package in `src/symptom_dx/`, run through the `symptom-dx` command.

## Commands

```bash
pip install -e ".[dev]"                 # into .venv
pytest                                  # all tests (offline, no keys needed)
pytest tests/test_pipeline.py::test_predict_skips_a_failing_source   # single test
ruff check src tests && ruff format src tests

symptom-dx preprocess | build-kg [--reset] | build-index | plot
symptom-dx predict "itching, skin rash" [--sources kg vector web] [--show-sources] [--verbose]
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check` and `pytest`.

## Architecture

- **Data flow:** `data/raw/symbipredict_2022.csv` (one-hot) → `preprocess` → `data/processed/preprocessed_data.csv` (`symptoms,prognosis`) and `symptoms_corpus.txt`. The graph is built from the CSV and the vector index (`artifacts/vector_index/`, gitignored) from the corpus.
- **Retrievers** (`knowledge_graph.py`, `vector_store.py`, `web_agent.py`) are small classes with a `name` and a `retrieve(symptoms) -> str` method. `pipeline.build_retrievers` imports each one only when that source is chosen, so tests and unused sources never touch Neo4j, Ollama or the network.
- **`pipeline.predict`** normalizes the input, runs each retriever (a failing source is recorded in `Prediction.errors` and skipped) and has the LLM merge the answers into `Prediction.report`. The evaluation's techniques 1–5 are source combinations listed in the README.
- **Symptom names** are canonical snake_case from `data.normalize_symptom`. Preprocessing, the graph and user input all go through it, so `"Skin Rash"` matches `skin_rash`. It also fixes source column typos such as `dischromic _patches`.
- **Graph model:** `(:Disease {name})-[:HAS_SYMPTOM]->(:Symptom {name})`, with uniqueness constraints on `name`. The load is one parameterized `UNWIND` query; never interpolate data into Cypher. `GRAPH_SCHEMA` and the few-shot examples in `prompts.py` must match this direction (a test checks it).
- **Config:** everything comes from environment variables or `.env` through `config.load_settings()`, which reads `.env` from the current folder. File paths are constants in `config.py`. Modules call `settings.require(...)`, which raises `MissingSettingError`; the command line prints it and exits with code 2.
- **Evaluation:** `results/metrics.csv` and `results/inference_time.csv` are long-format CSVs that `plots.py` reads. No evaluation code computes these numbers.

## Gotchas

- `.env` holds real credentials and must stay gitignored.
- The vector index stores text from the corpus it was built from. Rebuild it after changing preprocessing.
- `GROQ_MODEL` defaults to `openai/gpt-oss-120b` because Groq retired `llama-3.3-70b-versatile`. Groq's model list changes often.
- The web source needs the `ddgs` package (renamed from `duckduckgo-search`) and a real search region (`us-en`); the default `wt-wt` breaks its Wikipedia fallback.
- `KnowledgeGraphRetriever` sets `allow_dangerous_requests=True` because it runs LLM-written Cypher. Use a read-only Neo4j user for `predict`.
