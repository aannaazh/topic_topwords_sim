# TopWords Simulation (sim_parallel.py)

`sim_parallel.py` is the main end-to-end simulation and evaluation script for the TopWords topic model.
It supports synthetic corpus generation, EM training, vocabulary discovery, and multi-mode diagnostics/visualization.

## What this script does

1. **Generate synthetic data** with ground-truth topic labels, segmentation, and sentence-level mixture ratio `gt_pi`.
2. **Build model vocabulary** either:
   - `discover` (default): discover candidates from corpus n-gram frequency + filtering
   - `oracle`: use simulator ground-truth vocabulary (for upper-bound reference)
3. **Train TopWordsTopicModel** with parallel E-step style updates.
4. **Evaluate** with segmentation F1 and topic assignment accuracy (Hungarian alignment).
5. **Log + plot** for single run, distribution runs, and parameter comparison runs.

---

## Core modes

### 1) `single`
Runs one simulation and saves one consolidated diagnostics figure (`*.png`) with subplots:

- Log Likelihood curve
- F1 curve
- Topic Accuracy curve
- Vocabulary learning dynamics (mean topic entropy + effective vocab95)
- Phase portrait (F1 vs TopicAcc, colored by entropy)
- Topic confusion heatmap
- `pi` distribution (predicted vs ground-truth)
- Per-topic entropy trajectories
- Vocabulary precision/recall panel

Output:
- `results/plot/single/metrics_<timestamp>.png`
- `log/sim_<timestamp>.log`

### 2) `dist`
Runs multiple independent simulations and plots distribution statistics for F1 and TopicAcc.

Output:
- `results/plot/dist/dist_<timestamp>.png`
- `log/sim_<timestamp>.log`

### 3) `compare`
Runs grid-style experiments by varying one parameter (e.g. `num_topics`) and plots cross-setting comparisons.

Output:
- `results/plot/compare/compare_<target_param>_<timestamp>.png`
- `log/sim_<timestamp>.log`

---

## Key training/evaluation signals

`run_test()` summary includes:

- `final_f1`
- `final_topic_acc`
- `final_topic_entropy`
- `final_effective_vocab95`
- `final_vocab_precision_recall`
- `final_pi_mean`
- `final_pi_p10_p90`
- discovered vocabulary stats (`candidates`, `after_freq`, `selected`)
- vocabulary overlap with simulator truth

---

## Important parameters

### Data/model scale
- `--docs` (default 10000)
- `--num_topics` (default 10)
- `--char_size` (default 500)
- `--vocab_size` (default 1000)
- `--max_word_len` (default 4)
- `--sents_per_doc` (default 10)

### Vocabulary generation
- `--vocab_strategy {discover,oracle}` (default `discover`)
- `--vocab_min_freq` (default `2`)
- `--vocab_max_candidates` (default `0`, means unlimited)

### Optimization / stopping
- `--iterations` (default 50)
- `--tol` (default `0.0001`)
- `--min_iters_before_stop` (default `5`)
- `--early_stop_patience` (default `5`)

### Runtime
- `--workers` (default `8`)
- `--mode {single,dist,compare}`
- `--runs` for `dist`/`compare`
- `--target_param`, `--param_list` for `compare`

---

## Example commands

### Single run (1000 docs, 20 topics)
```bash
uv run python sim_parallel.py \
  --mode single \
  --docs 1000 \
  --num_topics 20 \
  --workers 8 \
  --iterations 10 \
  --tol 0.0001 \
  --min_iters_before_stop 5 \
  --early_stop_patience 5 \
  --alpha 0.1 \
  --beta 0.01 \
  --sents_per_doc 10 \
  --char_size 500 \
  --vocab_size 1000 \
  --max_word_len 4 \
  --single_char_ratio 0.3 \
  --vocab_strategy discover \
  --vocab_min_freq 2 \
  --vocab_max_candidates 1000
```

### Distribution run (variance check)
```bash
uv run python sim_parallel.py \
  --mode dist \
  --runs 10 \
  --docs 1000 \
  --num_topics 20 \
  --workers 8 \
  --iterations 10 \
  --tol 0.0001 \
  --min_iters_before_stop 5 \
  --early_stop_patience 5 \
  --vocab_strategy discover \
  --vocab_min_freq 2 \
  --vocab_max_candidates 1000
```

### Parameter comparison
```bash
uv run python sim_parallel.py \
  --mode compare \
  --target_param num_topics \
  --param_list 2,5,10,20 \
  --runs 10 \
  --docs 1000 \
  --workers 8
```

---

## Notes

- `discover` mode is intentionally harder than `oracle` mode and gives more realistic F1 behavior.
- Run-to-run variance can be non-trivial due to randomness in corpus sampling, vocabulary discovery overlap, and model initialization.
