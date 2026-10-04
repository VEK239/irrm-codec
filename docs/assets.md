# Data and model assets

The Git repository distributes code, small format examples, experiment configuration records and result tables. It does not currently distribute the locked dataset, trained model checkpoint, DATA-ANCHOR bundle or a DOI-backed model/data release. Their existing hashes and run locations are recorded in [results/paper/provenance_and_hashes.tsv](../results/paper/provenance_and_hashes.tsv); recorded cluster paths describe past runs and are not public download links.

## Prepared training dataset

A prepared benchmark directory supplies:

- `dataset.parquet`: rows with `row_index`, `junction_aa`, `split`, `log10_pgen`, `log10_pgen_1mm` and the metadata used by evaluation.
- `embeddings.npy`: TCRemP targets aligned with the table's row indices.
- Split and training-subset manifests produced by the dataset-preparation command.
- `target_standardizer.npz`: train-only normalization statistics, reused by all matched conditions.

The original inputs are an AIRR receptor table and TCRemP embeddings keyed by `clone_id`. Generate probability targets with the pinned mirpy/OLGA environment. Preserve IDs and row order; the preparation and preflight routines test alignment.

## Checkpoint bundle

Keep these files together when moving or releasing a run:

- `best.pt`: selected model weights plus the exact architecture and tokenizer metadata.
- Original tokenizer JSON and every file referenced by that JSON (for anchored tokenizers, include the middle WordPiece bundle).
- `target_standardizer.npz`: required for native target-space predictions and evaluation.
- `run_config.json`, metric tables, split manifests and hashes identifying the training experiment.

`rtp-codec encode --tokenizer ...` overrides a recorded tokenizer path while preserving vocabulary IDs. A tokenizer with the same vocabulary size but different token surfaces is not interchangeable. The inference API uses the dimension saved in the checkpoint, including older 320-dimensional configurations.

A complete public data/model release remains a separate step: archive the exact assets, verify hashes, attach their licenses and add stable download links here. Source installation and small CPU tests work without those assets; the full manuscript experiments require them.

Earlier 100,000-row AIRR files for IGH, IGK, IGL, TRD and TRG were removed from the current source tree; they remain available in Git history before the structure migration. They are historical cross-chain inputs, not the locked TRB manuscript dataset. The local migration also preserves them in `rtp-codec-legacy-airr-data.zip` with a SHA-256 manifest.
