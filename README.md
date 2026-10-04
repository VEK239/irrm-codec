# RTP-CODEC

![RTP-CODEC model: CDR3 sequence, anchor tokenizer, Transformer encoder and three biological objectives](docs/figures/model-schematic.png)

**Learning Compact TCR Representations from Multiple Biological Objectives**<br>
**Accepted to BioFM @ ICDM'26**

Elizaveta Vlasova · Petr Grigorev · Mariia Demidova · Sergey Muravyov · Mikhail Shugay

RTP-CODEC learns a **128-dimensional representation of a human TRB CDR3 amino-acid sequence**. It combines sequence reconstruction, TCRemP-derived receptor geometry and generation-probability prediction in a shared Transformer bottleneck. Biological targets are computed offline for training; the trained encoder requires only the CDR3 sequence, without TCRemP computation or V/J annotations at inference.

This repository provides the model, tokenizers, training and evaluation pipelines, matched ablation configurations and result tables accompanying the paper.

[Method](#method) · [Results](#main-results) · [Quick start](#quick-start) · [Reproduction](#reproduce-the-paper) · [Citation](#citation)

## Method

The **DATA-ANCHOR tokenizer** represents recurrent CDR3 terminal motifs with dedicated tokens and encodes the variable middle with WordPiece. A shared encoder produces one latent vector, supervised by three heads:

| Objective | Supervision |
| --- | --- |
| **R — reconstruction** | Recover the original CDR3 with an autoregressive character decoder |
| **T — TCRemP** | Predict standardized TCRemP reference coordinates using MSE and cosine loss |
| **P — generation probability** | Predict `log10_pgen_1mm`, the log probability of the sequence and its one-amino-acid neighborhood, using Huber loss |

The paper model uses a 128-D bottleneck, hidden width 320, eight attention heads and four encoder/four decoder layers. Input sequences have at most 40 residues. Target normalization and terminal motifs are fitted on training rows only; the middle WordPiece vocabulary uses an independent OLGA-generated corpus filtered against all benchmark splits. Reconstruction remains character-level for every encoder tokenizer.

The matched **R, T, P, RT, RP, TP and RTP** variants keep architecture and data fixed while changing active objectives. See [architecture and losses](docs/architecture.md) and [tokenizer construction](docs/anchored_tokenizer_study.md).

## Main results

On the held-out synthetic TRB benchmark, reconstruction and Pgen supervision are compatible. Adding TCRemP supervision changes the tradeoff between recovering the sequence, retaining generation information and matching receptor geometry:

| Objectives | Reconstruction exact match ↑ | TCRemP cosine ↑ | Pgen RMSE ↓ |
| --- | ---: | ---: | ---: |
| R | 0.9766 | — | — |
| RP | 0.9813 | — | 0.2264 |
| RTP | 0.9636 | 0.99190 | 0.2702 |

These are the codec's **native heads**; Pgen RMSE is measured for `log10_pgen_1mm`. The [complete seven-condition factorial](research/results/paper/factorial_results.tsv) includes the remaining objective combinations.

A separate comparison trains new heads on **frozen representations**. RTP-CODEC reaches reconstruction exact match **0.8831 ± 0.0058** and Pgen RMSE **0.2375 ± 0.0008** (mean ± standard deviation across probe seeds) in this setup, outperforming the evaluated TCRemP, TCR-BERT, ESM2-35M and CDR3-only SCEPTR representations on both recovery tasks. Reconstruction probes use a common 64-D comparison space. These probe results should be read separately from the native-head table above; see the [consolidated comparison](research/results/paper/external_author_global_comparison.tsv).

The paper additionally examines transfer to independent real VDJdb receptors, epitope geometry and inference runtime. All available tables, controls and error analyses are indexed in [paper results](research/results/paper/README.md).

## Quick start

Python 3.11+ is required. From a fresh checkout:

```bash
git clone https://github.com/VEK239/rtp-codec.git
cd rtp-codec
python -m pip install -e .
```

Encode a TSV containing a `junction_aa` column with a trained checkpoint and its original tokenizer bundle:

```bash
rtp-codec encode \
  --checkpoint /path/to/best.pt \
  --tokenizer /path/to/anchored_tokenizer.json \
  --input research/examples/data/sequences.tsv \
  --output artifacts/latents.npy \
  --device cpu
```

The output is a float32 NumPy matrix with one vector per input row in the same order, plus a JSON metadata file. Use `--device cuda` for GPU inference or `--sequence-column cdr3` for a differently named input column.

```python
from rtp_codec.inference import load_encoder

encoder = load_encoder(
    "/path/to/best.pt",
    tokenizer_path="/path/to/anchored_tokenizer.json",
)
vectors = encoder.encode_sequences(["CASSLGQETQYF", "CASSIRSSYEQYF"])
# Shape: (2, 128) for the paper checkpoint.
```

**Asset availability:** the repository contains source, configurations and result tables; pretrained weights, the DATA-ANCHOR bundle and the full locked dataset are not currently bundled or linked as a public release. Encoding requires those model assets. See [required files](docs/assets.md); installation variants and further API examples are in [usage](docs/usage.md).

## Reproduce the paper

All commands run from the repository root. The following stages cover the experiment workflow; the [full protocol](docs/reproduction.md) specifies preparation commands and external model requirements.

1. **Prepare and lock the inputs.** Use the original 99,430-receptor TRB benchmark, with 79,544/9,943/9,943 train/validation/test rows. The prepared directory contains `dataset.parquet`, aligned `embeddings.npy`, split manifests and `target_standardizer.npz`. Keep the complete DATA-ANCHOR bundle, including its middle WordPiece JSON. If preparing inputs from scratch, follow the [dataset and tokenizer recipe](docs/reproduction.md#1-prepare-and-lock-the-benchmark); changed source data or regenerated splits define a new experiment.
2. **Run the matched objective factorial.** Use the seven presets in [research/configs/paper](research/configs/paper), sharing the same data, tokenizer and train-only normalizer. Each preset uses batch size 64 and up to 40 epochs, with early stopping disabled and checkpoint selection by joint validation loss.
3. **Evaluate the saved representations.** Compare native heads on the held-out synthetic split, train fresh reconstruction/Pgen probe heads on frozen encoders, and evaluate transfer on the independent VDJdb cohort. Keep training-only projection fitting and the cohort overlap checks. Evaluation protocols and cluster launchers are linked below.
4. **Check and retain the outputs.** Each training run writes `best.pt`, `last.pt`, `run_config.json`, normalization statistics and `test_metrics.json`. Preserve asset hashes and preflight reports, then compare metrics with the [recorded paper tables](research/results/paper/README.md).

Install target-generation and test dependencies, then run the factorial (Bash):

```bash
python -m pip install -e ".[pgen,dev]"

for arm in r t p rt rp tp rtp; do
  rtp-codec train --config "research/configs/paper/$arm.json" \
    --data-dir data/benchmark/trb \
    --tokenizer-path data/tokenizers/data_anchor/anchored_tokenizer.json \
    --device cuda
done
```

For one model, run the same command with `--config research/configs/paper/rtp.json`. Explicit CLI options override preset values. External encoder comparisons additionally require the `benchmarks` extra and their model weights; TCRemP requires its separate software and prototype data.

| Experiment | Where to start |
| --- | --- |
| Dataset preparation, tokenizer audits and training | [Reproduction guide](docs/reproduction.md) |
| Frozen reconstruction and Pgen probes | [Benchmark protocols](docs/protocols/) and [evaluation modules](src/rtp_codec/benchmarks/) |
| VDJdb transfer and cohort checks | [VDJdb validation](docs/vdjdb_epitope_validation.md) |
| GPU jobs and preflight gates | [Cluster guide](docs/cluster.md) and [Slurm templates](research/scripts/slurm/) |

A small CPU check is available without the full benchmark assets:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests include a miniature training run, tokenization, data alignment and checkpoint-based inference. They validate the implementation; full paper metrics require the locked assets and complete experiment runs.

## Repository

| Directory | Contents |
| --- | --- |
| [`src/`](src/) | Installable `rtp_codec` library and CPU tests |
| [`research/`](research/README.md) | Paper configurations, results, launchers, examples and historical notebooks |
| [`docs/`](docs/README.md) | Usage, reproduction, architecture and migration guides |

The opening diagram is the original first figure of the final manuscript; its [vector PDF](docs/figures/model-schematic.pdf) is preserved.

## Citation

**Accepted to BioFM @ ICDM'26.**

```bibtex
@inproceedings{vlasova2026rtpcodec,
  title = {RTP-CODEC: Learning Compact TCR Representations from Multiple Biological Objectives},
  author = {Vlasova, Elizaveta and Grigorev, Petr and Demidova, Mariia and Muravyov, Sergey and Shugay, Mikhail},
  booktitle = {BioFM @ ICDM'26},
  year = {2026},
  note = {Accepted}
}
```

Software citation metadata: [CITATION.cff](CITATION.cff). Code license: [GPL v3](LICENSE).
