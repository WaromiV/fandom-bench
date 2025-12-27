## Fandom Bench

An async, CSV-driven LLM benchmarker built on the latest LangChain runnable APIs. It runs prompts against multiple providers and uses an OpenAI-based supervisor model (configured as “gpt-5” by default for forward compatibility) to grade answers.

### Features
- CSV input (`id,prompt,expected_answer` by default) with metadata passthrough for extra columns.
- Modular provider registry (OpenAI, Anthropic, Cohere, Mistral, Groq) and easy extension.
- Async execution with bounded concurrency and optional supervisor grading.
- Safe “no token” dry-run: missing API keys or optional dependencies are reported instead of crashing.

### Quickstart
1. Create/activate a virtualenv and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare a CSV (or use `sample_benchmark.csv`):
   ```bash
   python bench.py --csv sample_benchmark.csv \
     --provider openai:gpt-4o --provider anthropic:claude-3-5-sonnet \
     --supervisor-model gpt-5
   ```
   - Add environment variables for the providers you use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`).
   - The supervisor uses OpenAI. If `OPENAI_API_KEY` is absent, grading is skipped but generations still run.

### CLI flags
- `--csv PATH` (required): CSV with columns `id`, `prompt`, `expected_answer` (name overrideable).
- `--provider PROVIDER:MODEL[:NAME]`: multiple allowed. `NAME` is an optional label; defaults to `provider-model`.
- `--supervisor-model MODEL`: OpenAI model for grading (default `gpt-5` placeholder; swap to `gpt-4o` etc.).
- `--prompt-column`, `--expected-column`: override CSV field names.
- `--max-concurrency`: limit concurrent generations (default 3).
- `--max-records`: trim dataset for quick checks.
- `--temperature`: shared temperature for model calls (default 0.0).
- `--dry-run`: parse and build configs without invoking any model calls.

### Extending providers
See `fandom_bench/providers.py` and add a `ProviderDetails` entry with its env var, import path, and pip package name. The runner will report missing keys/deps cleanly.

### Sample CSV
`sample_benchmark.csv` includes three toy prompts for a quick sanity check. Running without tokens will show “skipped (missing_api_key)” statuses so you can verify wiring before adding secrets.
