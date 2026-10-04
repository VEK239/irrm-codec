# Migration from IRRM-CODEC

The GitHub repository is now `VEK239/rtp-codec`, matching the manuscript name. Repository history is preserved; no historical commits or reported result tables were rewritten.

Update a clone and install the package:

```bash
git remote set-url origin https://github.com/VEK239/rtp-codec.git
git pull
python -m pip install -e .
```

## Python and command paths

| Previous | Current |
| --- | --- |
| `irrm_codec.multitask_transformer` | `rtp_codec.models.codec` |
| `IRRMCodecConfig`, `IRRMCodecTransformer` | `RTPCodecConfig`, `RTPCodecTransformer` |
| `irrm_codec.train_multitask` | `rtp-codec train` / `rtp_codec.training.multitask` |
| `irrm_codec.multitask_data` | `rtp_codec.data.multitask` |
| `irrm_codec.tokenization` | `rtp_codec.tokenization.character` |
| `irrm_codec.wordpiece_tokenization` | `rtp_codec.tokenization.wordpiece` |
| `irrm_codec.anchored_tokenization` | `rtp_codec.tokenization.anchored` |
| `benchmark.<module>` | Task-specific module under `rtp_codec.benchmarks` |
| `scripts.<Python module>` | Task-specific module under `rtp_codec.experiments` |
| `experiments/trb/<recorded configuration>` | `configs/trb/<recorded configuration>` |
| `results/author` | `results/paper` |
| Root requirements files | `pyproject.toml` and `requirements/` |

Python modules are launched with `python -m ...` after installation. The old `irrm_codec` Python namespace is no longer exposed. Existing shell launchers have been updated; update imports in your own notebooks or external scripts.

## Existing model assets

Training checkpoints contain an architecture dictionary and a state dictionary, so the saved tensor keys and numerical architecture are preserved. Load the architecture from the checkpoint. The new default latent width is 128 to match the manuscript; older explicit 320-width checkpoints must retain their saved configuration.

Character/anchored token IDs are PAD=0, BOS=1, EOS=2, UNK=3. Earlier WordPiece bundles may use PAD=0, UNK=1, BOS=2, EOS=3; the loader continues to honor their saved IDs. Never substitute or regenerate a tokenizer when evaluating an existing checkpoint.

Recorded JSON configs, CSV/TSV results, checkpoint hashes and historical remote paths are retained as provenance. They may refer to the old repository name or cluster filesystem. Generated datasets and experiment run directories remain untracked; moving the source does not migrate those external assets.

Cluster launchers accept `RTP_CODEC_ROOT` and `RTP_CODEC_PYTHON` where applicable. See [cluster.md](cluster.md) for an explicit new-run template and [assets.md](assets.md) for required bundles.
