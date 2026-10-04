# Leakage-safe TRB WordPiece vocabulary sweep (Stage A)

Status: complete. Selection used only the full validation objective. Test metrics were
recorded after checkpoint selection and were not used to rank conditions.

## Locked controls

- Train-only tokenizer fit: 79,544 sequences; train manifest SHA-256
  `6399cbea503191a652e7ecef0b8b859793f4ece16322fd49ce2ed03364484b01`.
- Benchmark READY SHA-256:
  `ce08945c0c0a975c2b05129695c26399456f185fe79b89914c0ad52e4fdcf167`.
- Special IDs: PAD/BOS/EOS/UNK = 0/1/2/3; zero UNK and exact roundtrip on train,
  validation, and test for every vocabulary.
- Matched model: full R+T+P, seed 42, latent 128, encoder/decoder 4+4, 40 epochs,
  batch 64, AdamW 3e-4, locked train-only normalizer, separate WordPiece encoder and
  unchanged 25-token character decoder.

## Validation-only ranking

| Rank | WordPiece vocab | Job | Best epoch | Full val objective | Val pgen RMSE | Val recon loss | Val token acc | Val TCRemP cosine | Val TCRemP std MSE | Params | Walltime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44 | 1426073 | 30 | 0.196663 | 0.343658 | 0.004776 | 0.998500 | 0.992229 | 0.258091 | 21,157,033 | 48m11s |
| 2 | 256 | 1426123 | 23 | 0.206499 | 0.272951 | 0.009048 | 0.996913 | 0.991787 | 0.270087 | 21,224,873 | 49m14s |
| 3 | 64 | 1426074 | 28 | 0.206852 | 0.324221 | 0.007868 | 0.997243 | 0.991838 | 0.269134 | 21,163,433 | 51m00s |
| 4 | 128 | 1426122 | 26 | 0.209016 | 0.283852 | 0.009564 | 0.996720 | 0.991729 | 0.272229 | 21,183,913 | 46m54s |
| 5 | 512 | 1426176 | 25 | 0.210491 | 0.308438 | 0.008969 | 0.997031 | 0.991697 | 0.273659 | 21,306,793 | 47m08s |
| 6 | 1024 | 1426177 | 28 | 0.210722 | 0.316921 | 0.006642 | 0.997884 | 0.991593 | 0.276770 | 21,470,633 | 49m07s |
| 7 | 2048 | 1426180 | 24 | 0.212181 | 0.326911 | 0.011945 | 0.996184 | 0.991786 | 0.270806 | 21,798,313 | 47m01s |
| 8 | 4096 | 1426181 | 25 | 0.216567 | 0.411111 | 0.012061 | 0.995960 | 0.991807 | 0.269719 | 22,453,673 | 48m49s |

The preregistered selection is WordPiece-44. Its validation objective is 0.000274
(0.139%) lower than the matched character baseline (0.196937), a small single-seed
difference that requires replication before a confirmatory claim.

## Held-out test metrics (descriptive only)

| Vocab | Total | Pgen RMSE | Recon loss | Exact match | Length accuracy | Mean edit | TCRemP cosine | TCRemP std MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 0.195894 | 0.341319 | 0.005795 | 0.978980 | 0.990747 | 0.047068 | 0.992311 | 0.255408 |
| 64 | 0.205969 | 0.325752 | 0.009292 | 0.964397 | 0.985015 | 0.081867 | 0.991931 | 0.265650 |
| 128 | 0.207857 | 0.289369 | 0.010379 | 0.959871 | 0.977371 | 0.087298 | 0.991808 | 0.269085 |
| 256 | 0.203443 | 0.270520 | 0.009503 | 0.963693 | 0.982199 | 0.084783 | 0.991930 | 0.265257 |
| 512 | 0.207351 | 0.315414 | 0.009045 | 0.966610 | 0.981897 | 0.075128 | 0.991832 | 0.268639 |
| 1024 | 0.209171 | 0.315975 | 0.007251 | 0.972543 | 0.983405 | 0.059841 | 0.991686 | 0.273637 |
| 2048 | 0.210649 | 0.329018 | 0.012147 | 0.955245 | 0.974052 | 0.096047 | 0.991866 | 0.268131 |
| 4096 | 0.216030 | 0.419660 | 0.013340 | 0.951725 | 0.970331 | 0.107010 | 0.991912 | 0.266441 |

## Token interpretation

The winning 44-token tokenizer is effectively the smallest valid standard WordPiece
system: `C` is the start token in all 79,544 training sequences, and most continuation
pieces are single amino acids. `##F` is strongly end-enriched (mean normalized position
0.934; end log2 enrichment 3.378). Frequent pieces include `##S` (166,534 occurrences,
94.0% of sequences) and `##A` (118,254, 97.7%).

Train-only descriptive presence associations show lower `log10_pgen_1mm` for sequences
containing `##V` (mean difference -1.211; high-vs-low-quartile log2 enrichment -1.483),
`##W` (-1.267; -1.852), and internal `##C` (-1.342; -1.909). These are descriptive,
unadjusted associations, not biological or causal claims.

At vocabulary 64, multi-residue boundary pieces emerge (`CA`, `CAS`, `##YEQYF`,
`##TQYF`, `##EQFF`). At 256, the same boundary families remain common. At 4096,
the vocabulary fragments into much rarer context-specific pieces: `##YEQYF` occurs in
2.49% of train sequences and is positively associated with higher pgen
(mean difference +0.665; quartile log2 enrichment +1.005), while rare single-residue
fallbacks such as `##F` show strong negative associations. This pattern is consistent
with reduced statistical sharing at very large vocabularies and accompanies worse joint
validation objectives, but it does not establish a biological mechanism.

## Artifacts

- Tokenizers and audits:
  `/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-sweep-seed42`
- Model runs:
  `/home/evlasova/irrm-codec/experiments/trb/wp-sweep-v<V>-rtp-seed42`
- Each run contains `best.pt`, `last.pt`, `history.json`, `run_config.json`,
  `test_metrics.json`, `target_standardizer.npz`, and logs. All terminal artifact and
  failure scans passed.

No replication or additional architecture jobs were launched.
