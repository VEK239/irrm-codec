# Reproduction and compatibility

## Data and results

The repository contains code, tests, experiment configurations, and historical
student result tables. Large TCRemP embeddings, pretrained weights, checkpoints,
raw repertoires, and cluster output are external runtime artifacts. CPU tests use
synthetic fixtures; they do not regenerate the published research results.

The author's joint preparation (`rtp_codec.benchmarks.datasets.prepare_trb_joint`) validates biological
identity using AIRR, Pgen, and the TCRemP clone-ID sidecar and writer contract. The
student preparation (`rtp_codec.benchmarks.datasets.prepare_splits`) retains its distinct benchmark
workflow and V/J cleaning. Do not interchange split manifests between cohorts.
Normalize targets and fit tokenizer/PCA models using the prescribed training data.

## Token IDs and historical checkpoints

The author's character and anchored-tokenizer convention is PAD=0, BOS=1, EOS=2,
UNK=3. Student WordPiece files and the earlier VEK239 main may use PAD=0, UNK=1,
BOS=2, EOS=3. The integrated WordPiece loader accepts both known layouts, preserves
all stored IDs, and checks UNK/EOS using each tokenizer's own vocabulary. The
reconstruction decoder uses the author's character convention.

A character reconstruction checkpoint trained under the earlier VEK239 special-ID
order is not automatically interchangeable with the author's decoder convention.
Use its original revision or explicitly migrate special-token embedding/output
rows before evaluation. Do not relabel token IDs in existing tokenizer files.
Author DATA-ANCHOR configurations remain compatible with the author convention.

## Cluster examples

Slurm examples retain the partitions, accounts, data paths, and environments used
for their historical experiments. Inspect these before submission; #SBATCH paths
are literal and do not expand shell variables. Adapt them to your cluster and use
the Python module entrypoints with explicit CLI paths. Scripts that expose REPO,
SCRATCH, CONDA_ROOT, PYTHON_BIN, or RTP_CODEC_ROOT can be configured through those variables.
No cluster jobs are submitted by installing this repository or running its CPU tests.

## Validation scope

Run `python -m pytest -q` from the repository root. Tests cover reconstruction,
objective combinations, tokenizers, data identity/alignment, standardizers, metrics,
external benchmark bookkeeping, and VDJdb cohort rules. Optional external encoders
need requirements/benchmarks.txt plus model downloads; complete GPU training and
external-data experiments require separately provisioned resources.

Tab-separated result files preserve empty terminal fields. Trailing tabs in these
files represent missing values and are retained for table-schema fidelity.
