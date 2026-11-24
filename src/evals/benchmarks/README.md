# Categorization Benchmarks

This directory contains labeled benchmark datasets and prompt context files for
evaluating Sensemaker's categorization pipeline (`src/categorization_runner.py`)
via `src/evals/categorization_benchmark.py`.

## Wedding Music Categorization Benchmark (`wedding_music_100.csv`)

### Dataset Creation

`wedding_music_100.csv` is a synthetic 100-comment benchmark authored with
Gemini/Jetski to simulate a multi-participant discussion among 8 wedding
participants and planners debating reception music and entertainment. It is
structured into **3 top-level topics** and **7 distinct opinions (stances)**:

| Expected Topic | Expected Stance | Count | Expected Opinion |
| :--- | :--- | :---: | :--- |
| Dance Floor Music Genres | `house_music` | 20 | House music keeps the wedding dance floor energized with a modern, continuous groove. |
| Dance Floor Music Genres | `80s_pop` | 20 | 80s pop hits are the best wedding music because every guest knows the lyrics and sings along. |
| Dance Floor Music Genres | `disco` | 20 | Classic disco music brings joyful, funky rhythms that get both older and younger guests dancing together. |
| Reception Timeline and Atmosphere | `scheduled_mix` | 12 | The reception should start with classic hits for all ages before shifting to late-night club music. |
| Reception Timeline and Atmosphere | `dinner_acoustic` | 10 | Cocktail hour and dinner need mellow acoustic or jazz music so guests can talk comfortably before dancing starts. |
| DJ Mixing and Live Performance | `dj_beatmatching` | 10 | A skilled DJ who beatmatches transitions keeps momentum on the dance floor better than a static playlist. |
| DJ Mixing and Live Performance | `live_band_energy` | 8 | A live band with horns and vocals brings authentic stage energy that recorded tracks cannot match. |

When adding comments or building new benchmark datasets with the same schema,
keep each comment anchored to a clear ground-truth `(expected_topic,
expected_opinion)` pair and avoid reserved `categorization_runner` input column
names (`topic`, `topics`, `quote_with_brackets`, `quote_id`).

### Schema

`wedding_music_100.csv` uses 6 columns:

- `participant_id`: Unique statement identifier (`c001`–`c100`).
- `expected_topic`: Ground-truth top-level topic (used as a cluster ID in
  discovery modes and exported as `--topics` in `guided_topics`; see below).
- `expected_opinion`: Ground-truth opinion statement (used as a cluster ID in
  discovery modes and exported to `--topic_and_opinion_csv` in
  `guided_taxonomy`; see below).
- `expected_stance`: Short readable slug for reporting (e.g. `house_music`).
- `expected_keywords`: Semicolon-separated domain keywords for opinion-label
  sanity checks (see below).
- `survey_text`: Comment text scored/categorized by `categorization_runner`.

`wedding_music_context.md` contains **only** the conversation background passed
to `categorization_runner --additional_context_file`.

### How `expected_topic` and `expected_opinion` Are Used

Models running in unconstrained discovery (`learn_all`) will invent their own
phrasing for topics and opinions rather than reproducing the exact strings in
the CSV. The evaluator does **not** use embedding distance or an LLM judge to
compare predicted strings to `expected_topic` or `expected_opinion`. Instead,
these columns serve two distinct roles depending on the evaluation mode:

1. **Structural clustering evaluation (`learn_all` and `guided_topics`):**
   `expected_topic` partitions the 100 comments into **3 ground-truth
   clusters**, and `expected_opinion` partitions them into **7 ground-truth
   clusters**. Each distinct non-`Other` label predicted by the model is treated
   as a predicted cluster ID, and the evaluator compares the predicted
   partition against the ground-truth partition (separately at the topic and
   opinion levels) using label-agnostic clustering metrics:
   - **`Cluster Purity` (Precision):** For each predicted cluster, the fraction
     of its assigned comments belonging to a single dominant ground-truth
     group. Penalizes *over-merging* (e.g. mixing disco and house music into
     one opinion).
   - **`Inverse Purity (Recall)`:** For each ground-truth group, the fraction
     of its comments placed together in its single largest predicted cluster.
     Penalizes *over-splitting* and comments dropped into `Other`.
   - **`Purity F1`:** Harmonic mean of `Cluster Purity` and `Inverse Purity`.
   - **`Pairwise Co-assignment Precision / Recall / F1`:** Across all 4,950
     pairs of comments, checks whether two comments share a predicted cluster
     if and only if they share the same `expected_topic` / `expected_opinion`.
   - **`V-Measure` (`Homogeneity` & `Completeness`):** Information-theoretic
     harmonic mean (via `sklearn.metrics.homogeneity_completeness_v_measure`)
     balancing pure predicted clusters against complete recall of ground-truth
     groups.
2. **Taxonomy prompts and exact matching (`guided_topics` and
   `guided_taxonomy`):** The full human-readable strings in `expected_topic`
   and `expected_opinion` are exported directly as input taxonomy prompts for
   `categorization_runner` (`--topics` in `guided_topics`, and
   `--topic_and_opinion_csv` in `guided_taxonomy`). In those modes the model
   assigns comments to the supplied taxonomy, so `Exact Label Match Rate`
   (case-insensitive string match) measures direct classification accuracy
   (whereas it is expected to be `0%` in `learn_all`).

**How `expected_opinion` relates to `expected_keywords`:** Clustering against
`expected_opinion` verifies *which comments are grouped together* (structural
grouping), while `expected_keywords` verifies *that the recovered clusters were
given semantically sensible labels* (see below).

### How `expected_keywords` Is Used

`expected_keywords` is **not** used to evaluate quote extraction (the benchmark
defaults to `--skip_quote_extraction`). Instead, it pairs with the clustering
metrics above as a lightweight **semantic sanity check on the discovered
opinion labels** in unconstrained discovery (`learn_all`) and topic-guided
(`guided_topics`) modes:

- Because clustering metrics are completely invariant to cluster names, a model
  could group comments structurally well while emitting opaque or placeholder
  labels (e.g. `"Opinion 1"`).
- After clustering identifies the **dominant predicted cluster** for each
  `expected_opinion` group, the evaluator checks (case-insensitively) whether
  the **predicted `topic + opinion` label** contains at least one domain
  keyword from `expected_keywords`:
  - **`Statement Keyword Hit Rate`**: Fraction of the 100 ground-truth
    statements whose assigned non-`Other` `topic + opinion` label(s) match at
    least one expected keyword for that statement's ground-truth opinion.
  - **`Group Keyword Coverage`**: Fraction of the 7 ground-truth opinions whose
    dominant predicted `topic + opinion` cluster label matches at least one of
    its expected keywords.

### Relationship to LLM-as-a-Judge Autoraters (`src/evals/evals.py`)

This benchmark complements—rather than replaces—the existing reference-free
autoraters in `src/evals/evals.py` and `src/evals/run_all_evals.sh`
(`quote_extraction`, `opinion_categorization`, `opinion_quality`,
`other_opinion`):

- **Autoraters (`evals.py`)** require no labeled ground truth, work on any
  real-world survey, and are best suited for **local per-item quality**
  (faithful/concise quote extraction, whether a quote genuinely supports its
  assigned opinion, and opinion wording quality).
- **This benchmark (`categorization_benchmark.py`)** is **reference-based and
  deterministic** (no judge LLM cost or cross-model judge bias) and is best
  suited for **global taxonomy and clustering structure** (recovering all 7
  distinct stances without collapsing genres together, over-splitting, or
  dropping stances into `Other`), which pointwise per-quote autoraters cannot
  directly measure.

Both can be run on the same `categorization_runner` output directory when
comparing models end-to-end.

## Example Invocations

### Gemini Model

```bash
GEMINI_API_KEY="your-api-key" python -m src.evals.categorization_benchmark \
  --run_categorization \
  --model_name gemini-3.5-flash \
  --mode learn_all \
  --recategorize_other \
  --output_dir /tmp/wedding_bench_gemini
```

### Gemma 4 on GCE (e.g. `gemma4:26b`, `gemma4:31b`, or `gemma4:e4b`)

```bash
MODEL_NAME=gemma4:26b bash src/gce_ollama_server.sh up
python -m src.evals.categorization_benchmark \
  --run_categorization \
  --mode learn_all \
  --model_name gemma4:26b \
  --output_dir /tmp/wedding_bench_learn_all_gemma4-26b
bash src/gce_ollama_server.sh down
```

### Gemma 3 Locally on CPU

```bash
python -m src.evals.categorization_benchmark \
  --run_categorization \
  --mode learn_all \
  --model_name gemma3:270m \
  --output_dir /tmp/wedding_bench_learn_all_gemma3-270m
```

### Guided Modes & Offline Evaluation

```bash
# Topic-guided discovery (supply ground-truth topics, learn opinions)
python -m src.evals.categorization_benchmark \
  --run_categorization \
  --mode guided_topics \
  --model_name gemma4:26b \
  --output_dir /tmp/wedding_bench_guided_topics

# Full taxonomy guided (supply ground-truth topics and opinions)
python -m src.evals.categorization_benchmark \
  --run_categorization \
  --mode guided_taxonomy \
  --model_name gemma4:26b \
  --output_dir /tmp/wedding_bench_guided_taxonomy

# Re-evaluate an existing categorized_with_other.csv offline
python -m src.evals.categorization_benchmark \
  --categorized_csv /tmp/wedding_bench_gemini/categorized_with_other.csv
```
