## Fandom Bench

An async, CSV-driven LLM benchmarker built on the latest LangChain runnable APIs. It runs prompts against multiple providers and uses an OpenAI-based supervisor model (configured as “gpt-5” by default for forward compatibility) to grade answers.

# Fandom Knowledge Benchmark Report

**Benchmark name:** `fandom_bench`
**Run date:** 2025-12-27
**Evaluator:** `bench.py`
**Supervisor model:** `gpt-5-mini`
**Tested model:** `openai:gpt-4o-mini`
**Records evaluated:** 42
**Question format:** short-form fandom knowledge checks (explicit fandom stated)

---

## 1. Executive Summary

This benchmark evaluates **fandom-specific knowledge recall** across literature, games, anime, film, and TV franchises.
The goal is to measure **canonical accuracy**, not reasoning or creativity.

**Key finding:**
`openai:gpt-4o-mini` demonstrates **strong general fandom knowledge** but exhibits **systematic weaknesses** in:

* niche or emotionally symbolic fandoms
* lore requiring precise terminology
* cases where multiple canon-adjacent concepts exist

---

## 2. Overall Performance

| Metric                           | Value     |
| -------------------------------- | --------- |
| Total questions                  | 42        |
| Correct                          | 36        |
| Partial                          | 1         |
| Incorrect                        | 5         |
| Strict accuracy                  | **85.7%** |
| Lenient accuracy (partial = 0.5) | **86.9%** |

Latency was stable and within acceptable bounds; performance analysis focuses solely on correctness.

---

## 3. Incorrect Responses (Canonical Errors)

The following failures represent **true knowledge gaps or hallucinations**, not grading artifacts.

### Undertale

* **Determination color**

  * Model answered: *Blue*
  * Canonical answer: **Red**
  * Error type: symbolic conflation (game mechanic vs lore)

### OMORI (major weakness)

* **OMORI’s knife**

  * Model answered: *The Knife*
  * Canonical answer: **Steak Knife**
* **White Space companion**

  * Model answered: *Mori*
  * Canonical answer: **Mewo**
  * Error type: name invention / fandom drift

### Dune

* **Life-extending substance**

  * Model answered: *Water of Life*
  * Canonical answer: **Spice**
  * Error type: category collision between related canon concepts

### Doctor Who

* **Doctor’s recurring French phrase**

  * Model answered: *Bonjour, mes amis*
  * Canonical answer: **Allons-y**
  * Error type: era bleed / paraphrase hallucination

---

## 4. Partial Credit Case

### Death Note

* **Rule affecting human lifespan**

  * Model provided a high-level explanation but failed to state the specific canonical rule cleanly.
  * Scored as *partial* due to correct theme but imprecise formulation.

---

## 5. Accuracy by Fandom (High-Level)

| Fandom                  | Accuracy | Notes                               |
| ----------------------- | -------- | ----------------------------------- |
| Harry Potter            | 100%     | Well-represented, low ambiguity     |
| The Lord of the Rings   | 100%     | Stable canonical recall             |
| A Song of Ice and Fire  | 100%     | Strong terminology                  |
| Undertale               | 67%      | Symbolic confusion                  |
| **OMORI**               | **33%**  | Weak grounding, name hallucinations |
| Minecraft               | 100%     | Robust factual recall               |
| The Stanley Parable     | 100%     | Narrative memory intact             |
| Neon Genesis Evangelion | 100%     | Consistent canon                    |
| Death Note              | 83%      | Rule precision weakness             |
| Dune                    | 67%      | Concept overlap errors              |
| Doctor Who              | 67%      | Era-specific knowledge instability  |

---

## 6. Observed Failure Patterns

Across incorrect answers, the following patterns emerge:

1. **Plausible hallucinations**
   The model invents believable but non-canonical terms when confidence is low.

2. **Concept collision**
   Closely related canon elements are substituted for one another (e.g. *Spice* vs *Water of Life*).

3. **Symbolic drift**
   Emotional or thematic associations override explicit lore facts.

4. **Era bleed (long-running franchises)**
   Knowledge from adjacent eras or interpretations contaminates precise answers.

---

## 7. Conclusions

`openai:gpt-4o-mini` performs well on **mainstream and structurally simple fandoms**, but reliability drops on:

* niche games
* emotionally symbolic narratives
* franchises with dense or overlapping lore

This benchmark successfully differentiates **true fandom knowledge** from **confidence-weighted guessing**, validating its usefulness as a cultural recall probe rather than a general QA test.

---

## 8. Recommended Next Steps

* Expand OMORI / Undertale / Dune question sets (high signal)
* Add adversarial “false-friend” questions
* Track verbosity vs correctness correlation
* Run cross-model comparisons using identical CSVs

---

**Status:** benchmark behaving as intended
**Confidence in results:** high

---

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
