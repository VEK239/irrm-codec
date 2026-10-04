# Usage and experiment details

**Learning Compact TCR Representations from Multiple Biological Objectives**

Elizaveta Vlasova, Petr Grigorev, Mariia Demidova, Sergey Muravyov and Mikhail Shugay

RTP-CODEC learns a **128-dimensional representation of a human TRB CDR3 amino-acid sequence**. A shared Transformer encoder is trained jointly to reconstruct the sequence (**R**), predict TCRemP reference coordinates (**T**), and estimate generation probability (**P**). Once trained, the encoder takes only the CDR3 sequence; TCRemP computation and V/J annotations are needed for training supervision, not for sequence encoding.

This repository contains the model, training pipeline, tokenizers, evaluation tools and result tables accompanying the manuscript *RTP-CODEC: Learning Compact TCR Representations from Multiple Biological Objectives*.


The three heads provide supervision during training. The reusable representation is the encoder output before those heads.

## Installation

Python 3.11 or newer is required. From the repository root:

```bash
git clone https://github.com/VEK239/rtp-codec.git
cd rtp-codec
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For a CPU-only environment, install CPU PyTorch before the package:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

Optional dependencies are grouped by purpose:

| Extra | Purpose |
| --- | --- |
| `pgen` | OLGA-backed target generation through the pinned mirpy revision |
| `benchmarks` | ESM, TCR-BERT and SCEPTR representation comparisons |
| `tracking` | Weights & Biases logging for earlier training workflows |
| `notebooks` | Interactive analyses and plots |
| `dev` | CPU test suite |

For example, `python -m pip install -e ".[pgen,benchmarks,dev]"`. External encoders also require their own model weights; TCRemP requires its separate installation and prototype data. The reported manuscript environment used PyTorch 2.4.1; GPU work additionally requires a compatible CUDA installation.

## Encode sequences

The repository contains source code and result tables. **Trained model weights, tokenizer bundles and the full benchmark data are not bundled here.** Encoding requires a compatible `best.pt` checkpoint and, for DATA-ANCHOR or WordPiece models, its original tokenizer bundle. See [data and model assets](../docs/assets.md).

Prepare a TSV with a `junction_aa` column; [research/examples/data/sequences.tsv](../research/examples/data/sequences.tsv) shows the format. Each sequence must contain standard amino acids and fit the checkpoint's maximum length (40 residues in the manuscript).

```bash
rtp-codec encode \
  --checkpoint artifacts/paper/rtp-seed42/best.pt \
  --tokenizer data/tokenizers/data_anchor/anchored_tokenizer.json \
  --input research/examples/data/sequences.tsv \
  --output artifacts/example_latents.npy \
  --device cpu
```

The output is a float32 NumPy matrix with one vector per input row in the same order, plus a JSON sidecar describing the input and matrix. The vector dimension comes from the checkpoint. `--tokenizer` relocates a saved bundle; it does not retrain or replace the tokenizer. Use `--sequence-column cdr3` for a differently named input column.

The same interface is available in Python:

```python
from rtp_codec.inference import load_encoder

encoder = load_encoder(
    "artifacts/paper/rtp-seed42/best.pt",
    tokenizer_path="data/tokenizers/data_anchor/anchored_tokenizer.json",
    device="cpu",
)
vectors = encoder.encode_sequences(["CASSLGQETQYF", "CASSIRSSYEQYF"])
# vectors.shape == (2, 128) for the manuscript checkpoint
```

## Train the manuscript model

The paper uses a locked synthetic TRB dataset of 99,430 unique receptors, split into 79,544 training, 9,943 validation and 9,943 test rows. DATA-ANCHOR edge motifs and target normalizers are fitted on training rows only. Its middle WordPiece vocabulary is trained on a separate 500,000-sequence OLGA corpus with all benchmark sequences excluded. Reconstruction always uses an autoregressive character decoder.

Seven matched objective presets live in [research/configs/paper](../research/configs/paper): `r`, `t`, `p`, `rt`, `rp`, `tp` and `rtp`. Supply the prepared dataset and tokenizer paths:

```bash
rtp-codec train \
  --config research/configs/paper/rtp.json \
  --data-dir data/benchmark/trb \
  --tokenizer-path data/tokenizers/data_anchor/anchored_tokenizer.json \
  --device cuda
```

These presets use a 128-dimensional bottleneck, hidden width 320, eight attention heads, four encoder and four decoder layers, batch size 64 and up to 40 epochs. Early stopping is disabled; the checkpoint with the lowest joint validation loss is retained. Command-line arguments override preset values. Training writes `best.pt`, `last.pt`, target normalization statistics, run configuration and evaluation metrics to the configured output directory.

For data preparation, tokenizer construction, the full objective factorial and downstream evaluation, follow [the reproduction guide](../docs/reproduction.md). Cluster launchers are described in [docs/cluster.md](../docs/cluster.md). A new split or regenerated tokenizer constitutes a new experiment; it should not be presented as an exact reproduction of the locked manuscript runs.

## Results reported in the manuscript

Native heads on the held-out synthetic benchmark:

| Objectives | Reconstruction exact match ↑ | TCRemP cosine ↑ | Pgen RMSE ↓ |
| --- | ---: | ---: | ---: |
| R | 0.9766 | — | — |
| T | — | 0.99149 | — |
| P | — | — | 0.2753 |
| RT | 0.9632 | 0.99178 | — |
| RP | 0.9813 | — | 0.2264 |
| TP | — | 0.99169 | 0.3496 |
| RTP | 0.9636 | 0.99190 | 0.2702 |

Source: [matched objective factorial](../research/results/paper/factorial_results.tsv). Pgen means regression of `log10_pgen_1mm`. Dashes indicate inactive objectives. These results show an objective tradeoff: adding T to RP improves teacher agreement while reducing native reconstruction and Pgen performance.

New heads trained on frozen representations provide a separate information-recovery comparison:

| Frozen representation | Reconstruction exact match ↑ | Pgen RMSE ↓ |
| --- | ---: | ---: |
| RTP-CODEC | 0.8831 ± 0.0058 | 0.2375 ± 0.0008 |
| TCRemP | 0.5181 ± 0.0024 | 0.6038 ± 0.0019 |
| TCR-BERT | 0.4778 ± 0.0179 | 0.4299 ± 0.0022 |
| ESM2-35M | 0.2042 ± 0.0132 | 0.7086 ± 0.0033 |
| SCEPTR, CDR3-only | 0.8213 ± 0.0043 | 0.6637 ± 0.0062 |

Source: [consolidated frozen-representation comparison](../research/results/paper/external_author_global_comparison.tsv). Values are mean ± standard deviation across probe seeds; reconstruction uses the common 64-dimensional comparison space. These are fresh probe heads, not the codec's native heads above.

On the independent real VDJdb cohort, RTP-CODEC achieved macro Cohen's d of 0.6866 and improved epitope separation over RP for 31 of 40 epitopes. Native TCRemP and its PCA-128 control remained stronger (0.9563 and 0.8230). The manuscript's single runtime measurements on 68,302 receptors were 29.26 s for RTP-CODEC on CPU, 9.36 s on GPU and 350.25 s for 16-thread TCRemP. Timings exclude loading and warm-up and use different pipeline boundaries; they are hardware-specific, not a general speed guarantee.

The [paper result index](../research/results/paper/README.md) links architecture, tokenizer, downstream and error-analysis tables with their provenance. Earlier benchmark tables are retained in [research/results/legacy](../research/results/legacy).

## Repository layout

The repository root contains `src/`, `research/` and `docs/`, plus package metadata, citation and license files. The library and CPU tests are under `src`; configurations, results, launchers, examples, environments and historical notebooks are grouped under `research`. See [the research index](../research/README.md).

Run `python -m pytest -q` after installing the `dev` extra. GitHub Actions also checks the public CLI and shell syntax.

## Citation and license

Please cite *RTP-CODEC: Learning Compact TCR Representations from Multiple Biological Objectives* by Elizaveta Vlasova, Petr Grigorev, Mariia Demidova, Sergey Muravyov and Mikhail Shugay. Accepted to BioFM @ ICDM'26. [CITATION.cff](../CITATION.cff) provides software citation metadata.

The code is distributed under [GNU GPL v3](../LICENSE). External models and datasets retain their own licenses.
