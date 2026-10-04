# Integration provenance

Canonical destination: https://github.com/VEK239/irrm-codec

The publication integration starts at Antigenomics main `31ccbad1c83a765fdb91a3686cbe31d4fcf6165c`.
Real merge commits preserve the original authors, commit IDs, and parent history:

1. Maria Demidova's `demidovamaria/main` at `7b0e7f1b82c64490528163210d154fddcb97a290` (45 unique commits).
2. `Consoooomer/benchmark/issue-3-pgen` at `b96f06eb42d9c92e3982103bb903a1dc67d59138` (19 unique commits), including the splits and reconstruction branches.
3. VEK239 main at `aa2589d21428fd59fd34b55d675fc2cb62a6980b`.
4. Antigenomics `agent/multitask-tokenizer-evaluation` at `9abb43c99de96108e178b8bbb5a0ed5d5cf414c6`, authored by Elizaveta Vlasova.
5. Previously uncommitted local author source, test, Slurm, and experiment configuration files.

Published history is not rewritten and student work is not squashed. The integration
retains both the published VEK239 main and student tips as ancestors. Contributor
identity follows the original Git commits; no author fields were reassigned.

Resolution decisions:

- Keep the working single-task training fixes and W&B integration from VEK239 main.
- Combine memory-mapped/clone-ID-aware embedding preparation with the student's V/J
  standardization and sequence-keyed Pgen cache.
- Retain student padded/anchored WordPiece APIs together with validated unpadded
  WordPiece input and the author's anchored tokenizer implementation.
- Preserve both known WordPiece special-token layouts without renumbering stored vocabularies.
- Retain author gradient-accumulation and resume fixes alongside the experimental
  locked standardizer and local TCRemP geometry-loss options.
- Retain historical reports and result tables with their original provenance.

The original GPL-3.0 license remains in LICENSE. Historical cluster paths and input
hashes in experiment manifests are provenance records and are not rewritten.
