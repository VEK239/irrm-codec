# Frozen residue-state benchmark preregistration

## Question

Does the poor reconstruction of public encoders in the common 64D sequence-vector
benchmark arise primarily from global pooling?  The corrective analysis exposes frozen
per-residue states to a trainable, position-aware task head.

## Locked inputs

- Cohort: `/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/cohort`
- Split: the existing 79,544 / 9,943 / 9,943 train/validation/test manifests.
- Models: ESM2-8M, `wukevin/tcr-bert@ef65ddc`, and SCEPTR 1.2.0 default.
- Encoder parameters are frozen.  Validation loss alone chooses checkpoints.
- `log10_pgen_1mm` is normalized using the training split only.

## Representation interfaces

The published pooled/vector results remain unchanged and are the compact-vector panel.
The new sensitivity panel uses:

- ESM2: final-layer amino-acid states, excluding BOS/EOS/padding.
- TCR-BERT: final-layer amino-acid states, excluding CLS/SEP/padding.
- SCEPTR: default model penultimate-layer CDR3B residue states selected with compartment
  mask 6.  The existing default SCEPTR vector is a learned CLS representation, not a
  mean-pooled vector.

States are zero-padded to 40 residues and stored as float16; task heads cast to float32.
No amino-acid states are averaged before reaching the trainable head.

## Task heads

Each representation gets its own freshly initialized decoder/probe for seeds 42, 43,
and 44.  Reconstruction uses the same learned output queries and three-layer
cross-attention decoder for every representation.  Pgen uses the same two-layer
position-aware attention probe for every representation.  Only the unavoidable input
projection width changes with the frozen encoder dimension; parameter counts are
reported.  All heads use AdamW at `3e-4`, matching the locked RTP training rate.  A
discarded `1e-3` pilot was stopped after its TCR-BERT validation loss diverged; no
pilot checkpoint or test result enters this comparison.

## Endpoints

- Reconstruction primary: exact CDR3 match.  Secondary: residue accuracy, normalized
  Levenshtein distance, edit distance <=1, and mean edit distance.
- Pgen primary: raw-unit RMSE.  Secondary: MAE, R2, Pearson, Spearman, and bias.
- The prespecified comparison is within encoder: residue-state head versus its existing
  pooled/vector head.  RTP remains the fixed compact-vector reference.

## Interpretation boundary

This is a best-available-frozen-interface sensitivity analysis, not a matched compact
representation comparison.  A residue tensor contains more numbers than a 64D or 128D
vector, and its task head is position-aware.  If it closes the gap, the pooled benchmark
was interface-limited.  If RTP remains better, that is stronger evidence for accessible
task information, but still not universal representation superiority because RTP was
trained on the benchmark reconstruction and pgen objectives.
