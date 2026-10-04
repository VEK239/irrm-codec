# VDJdb epitope-separation validation gate

This study asks whether frozen IRRM-CODEC latent vectors separate epitope/MHC
labels: within-label distances must be lower than matched between-label
distances. It is a quantitative held-out validation, not a visualization study.

## Cohort rule (locked before embeddings)

- Source: official VDJdb `2025-07-30` `vdjdb.txt`; the CPU preflight records its
  byte size and SHA-256.
- Retain human TRB rows with canonical CDR3 amino acids (length at most 40),
  `cdr3fix.good=true` without a sequence-changing fix, VDJdb score at least 1,
  a peptide, MHC-A and MHC class, and a subject identifier.
- The label is the exact normalized tuple `(epitope, mhc.a, mhc.b, mhc.class)`.
  This avoids pooling the same peptide across distinct MHC contexts.
- Donors are keyed by `(study-or-reference, subject.id)` so generic identifiers
  such as `donor1` do not collide across studies.
- Deduplicate exact CDR3s. Remove CDR3s assigned to multiple labels, CDR3s
  observed under multiple donors (for unambiguous donor-cluster inference), and
  overlaps with any locked benchmark train, validation, or test split.
- Primary cohort: every label with at least 50 unique remaining CDR3s and at
  least two donors. Report fixed threshold sensitivities at 25, 50, and 100.

The CPU job writes the immutable candidate table, per-label counts, overlap and
filter counts, threshold sensitivity, hashes, and a finite two-sequence CPU
smoke encoding. It explicitly does not extract the full embedding matrix.

## Model-comparison gate

The completed DATA-ANCHOR run is R+T+P only. The required primary matrix is
matched DATA-ANCHOR R-only, P-only, R+P, and R+T+P encoders, with the same train
data, tokenizer bundle, seed, architecture, optimizer, batch size, 40-epoch
budget, and validation-only checkpoint selection. R+P versus R+T+P is the direct
test of TCRemP's incremental value conditional on reconstruction and pgen.
T-only and R+T are useful optional attribution conditions, but are not required
for this primary comparison. The current multi-task implementation always
constructs the shared encoder, decoder, TCRemP head, and pgen head. Therefore
zeroing unused loss weights yields checkpoint-compatible P-only encoders; its
decoder and TCRemP head remain present but intentionally untrained, and their
metrics are not interpretable.

A separate, narrower auxiliary-head analysis uses R-only, R+T, R+P, and R+T+P.
The existing character-tokenizer quartet can answer that question for the char
condition, but cannot be mixed with DATA-ANCHOR R+T+P: tokenizer vocabulary,
input embeddings, latent width, and objective would be confounded.

No embeddings or outcome metrics are produced until the cohort and matched
model-comparison matrix are accepted.

## Future metrics (preregistered)

Primary: per-label within/between cosine-distance ratio (lower is better), with
length-matched between-label pairs. Secondary: Euclidean contrast, same-label
pair AUROC/AUPRC, donor-excluded precision@1/5/10, and MAP@R. Controls are an
absolute length-difference score and 1,000 label permutations within length and
study/donor strata. Uncertainty uses 2,000 donor-cluster bootstrap replicates.
All compared checkpoints use the same cohort, pair manifest, and resamples.
UMAP is not a primary endpoint and no scaler or metric learner is fit on these
held-out records.
