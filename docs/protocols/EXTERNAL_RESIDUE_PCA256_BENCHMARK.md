# Frozen residue-state PCA-256 benchmark preregistration

## Question and correction

The full `[40,D]` residue tensors showed that ESM2 and TCR-BERT retain near-exact
sequence identity, but give each clonotype far more information than a compact IRRM
embedding.  That diagnostic array was stopped and is excluded from comparative
results.  The corrected comparison maps every public encoder to exactly one 256D
point before any supervised head is fitted.

## Leakage-safe representation

For ESM2-8M, TCR-BERT and SCEPTR, the zero-padded residue states are flattened in
sequence order.  Per-coordinate mean and scale and a seeded randomized 256-component
PCA are fitted on the locked 79,544-row training manifest only.  The fixed transform
is then applied to train, validation and test.  Reports record source/projection
hashes, the train-manifest and row-index hashes, solver seed, explained variance,
shapes and finiteness.  Validation and test rows are never used to fit scaling or PCA.

## Downstream heads

Each encoder receives its own freshly initialized reconstruction decoder and pgen MLP
for seeds 42, 43 and 44.  The architecture, optimizer, early stopping and training
budget are the established compact-vector benchmark implementations; only the 256D
input values differ.  Checkpoints are selected on validation loss.  Reconstruction is
scored by exact match, residue accuracy and edit-distance measures.  Pgen prediction
is scored in raw `log10_pgen_1mm` units by RMSE, MAE, R2, Pearson, Spearman and bias.

## Interpretation boundary

This asks how much useful information can be retained in a common 256D linear
compression of each frozen residue representation.  It does not prove that the public
model's native pooling is poor in every application, and 256D remains twice the
selected RTP latent width of 128.  Existing native-vector/PCA-64 results and the new
residue-PCA-256 results are reported as distinct interfaces.
