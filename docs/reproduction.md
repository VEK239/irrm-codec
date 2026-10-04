# Reproducing the manuscript experiments

All commands below run from the repository root after `python -m pip install -e ".[pgen,dev]"`. Full experiments require the assets described in [assets.md](assets.md), adequate memory for TCRemP targets and a compatible GPU environment. Use `--help` on each module for its full options.

## 1. Prepare and lock the benchmark

```bash
python -m rtp_codec.benchmarks.datasets.prepare_splits \
  --airr-path data/raw/trb_background.tsv \
  --embeddings-path data/raw/trb_background_embeddings.parquet \
  --output-dir data/benchmark/trb \
  --chain TRB --locus beta --seed 42
```

This cleans receptors, matches embeddings by receptor identity, computes/caches probability targets and writes reproducible splits and nested training subsets. Dataset-dependent filtering can change row counts. For the manuscript's exact 99,430-row cohort, use the locked assets and verify their hashes; regenerating from different source data is a new benchmark.

## 2. Construct the DATA-ANCHOR tokenizer

```bash
python -m rtp_codec.experiments.tokenizers.prepare_anchored_tokenizers \
  --data-dir data/benchmark/trb \
  --output-root data/tokenizers \
  --external-corpus data/raw/independent_olga_trb.tsv \
  --generation-script /path/to/olga/generate_sequences.py \
  --olga-model-dir /path/to/olga/human_TRB_model \
  --external-rows 500000 --seed 42
```

The routine fits edge motifs on the training split, filters the independent corpus against the benchmark and emits tokenizer bundles and audit reports. Keep the DATA-ANCHOR JSON together with its referenced WordPiece JSON. Consult [the tokenizer study](anchored_tokenizer_study.md) for the alternative character, WordPiece and germline-anchor experiments. Exact replication also requires the original OLGA model and corpus identities.

## 3. Train matched objective combinations

For one condition:

```bash
rtp-codec train --config configs/paper/rtp.json \
  --data-dir data/benchmark/trb \
  --tokenizer-path data/tokenizers/data_anchor/anchored_tokenizer.json \
  --device cuda
```

For all seven conditions (Bash):

```bash
for arm in r t p rt rp tp rtp; do
  rtp-codec train --config "configs/paper/$arm.json"   \
    --data-dir data/benchmark/trb   \
    --tokenizer-path data/tokenizers/data_anchor/anchored_tokenizer.json   \
    --device cuda
done
```

Explicit CLI options override preset fields. Preserve the same data, tokenizer and standardizer for all arms. A preset is a readable launch configuration; the trainer's saved `run_config.json` records the resolved architecture, arguments and runtime.

## 4. Frozen representations and downstream tests

The installed evaluation modules are organized by task:

| Package | Work |
| --- | --- |
| `rtp_codec.benchmarks.representations` | Cache encoders, fit train-only projections, validate representation bundles |
| `rtp_codec.benchmarks.reconstruction` | Train and collect reconstruction probe heads |
| `rtp_codec.benchmarks.pgen` | Train and collect Pgen probe heads |
| `rtp_codec.benchmarks.downstream` | Prepare latent matrices and evaluate VDJdb geometry, retrieval and labels |
| `rtp_codec.benchmarks.analysis` | Sufficiency, substitution, length and position analyses |
| `rtp_codec.benchmarks.runtime` | Tokenization and encoder timing |
| `rtp_codec.benchmarks.reporting` | Assemble checked result summaries |

For example:

```bash
python -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_frozen --help
python -m rtp_codec.benchmarks.representations.preflight_author_global_embeddings --help
python -m rtp_codec.benchmarks.runtime.benchmark_rtp_runtime --help
```

The [protocol documents](protocols/) and grouped [Slurm templates](../scripts/slurm/) specify the respective asset layouts and experiment arguments. Use the final consolidated [frozen comparison](../results/paper/external_author_global_comparison.tsv) for the article's main ESM2-35M and CDR3-only SCEPTR comparison; earlier 8M and annotation-aware controls answer different questions.

## 5. Validate before large runs

```bash
python -m pytest -q
rtp-codec train --help
rtp-codec encode --help
```

The CPU suite validates code and small fixtures. It does not rerun the full 40-epoch factorial, download external model weights or reproduce manuscript-level numerical results. Keep preflight reports, output tables and asset hashes with every full experiment.
