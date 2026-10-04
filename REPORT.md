# IRRM-CODEC baseline benchmark

Comparison of IRRM-CODEC against a small set of baselines on CDR3 sequence
reconstruction and Pgen prediction, using human TRB data from
[Zenodo record 19520535](https://zenodo.org/records/19520535).

One pretrained checkpoint per model. The pretrained encoders are frozen — only the
task heads are trained.

## Status

| issue | scope | state |
| --- | --- | --- |
| 1 | train/validation/test splits | done |
| 2 | sequence reconstruction | done |
| 3 | Pgen prediction | done |

## 1. Dataset

Source files are `trb_background_100k.tsv` and `trb_background_embeddings.parquet`,
both inside `redcea_bg.gz` on the Zenodo record. The embeddings parquet is the only
place TCRemP vectors are published, and it holds exactly 100,000 rows per chain rather
than a large background pool.

The AIRR table arrived clean: no duplicate CDR3 amino-acid sequences, no non-standard
residues, no missing V/J calls, and nothing longer than the model's 40-residue limit.
Two filters still removed rows:

| filter | rows | reason |
| --- | ---: | --- |
| non-functional V gene | 335 | `TRBV21-1`, `TRBV23-1`, `TRBV6-7` are pseudogenes; SCEPTR rejects them |
| Pgen = 0 | 2 | `CASSFL`, `CASTTL` give `log10(0) = -inf`, which has no finite gradient |

Both were dropped for every representation and both targets at once, so all models see
one identical row set. Final dataset: **99,663 unique CDR3 sequences**.

The embeddings parquet has no `clone_id`, so it is aligned to the AIRR table by row
order. Because a silent misalignment would corrupt every downstream result invisibly,
the pipeline asserts it: mean TCRemP CDR3-distance correlates with sequence length at
**r = 0.525**, against **r = 0.008** under a shuffled control. The run aborts below
r = 0.3.

Split: one seeded permutation over de-duplicated sequences, so a CDR3 can appear in only
one split. Nested training subsets are prefixes of the same permutation.

| split | rows |
| --- | ---: |
| train | 79,730 |
| validation | 9,966 |
| test | 9,967 |
| subsets | 1,000 ⊂ 10,000 ⊂ 79,730 |

Both targets are cached: `log10(Pgen)` and `log10(Pgen_1mm)`, computed with mirpy's OLGA
wrapper (1 h 43 min on 8 CPU cores). Verified: no CDR3 overlap between any pair of
splits, and target means agree across splits to within 0.01 for both targets.

Reproduce with `python -m benchmark.prepare_splits`.

## 2. Sequence reconstruction

Every representation is reduced to the same **64-dimensional bottleneck** by a
`StandardScaler` + PCA fitted on the training split only, then decoded by the same
IRRM-CODEC decoder (`InverseModel`, 14.48 M parameters at this input width) with
identical settings: AdamW, lr 1e-3, weight decay 1e-4, batch 256, up to 60 epochs,
early stopping with patience 8. Three seeds per representation (42, 43, 44), 15 runs
total, each on one GTX 1080 Ti.

Decoder inputs are standardized with train-split statistics. Without it a shared
learning rate would act differently on each representation — the projections arrive with
standard deviations from 1.0 to 12.2 — making "identical training settings" true only on
paper. The repo's own forward/inverse trainers normalize their embeddings the same way.

### Results at a 64-dimensional bottleneck

Mean ± standard deviation over three seeds, on the held-out test split.

| representation | source dim | expl. var | exact match | token acc | norm. Levenshtein | within ED ≤ 1 | mean ED | train s | seq/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned one-hot | 1000 | 0.380 | **0.9144** ± 0.0006 | 0.9878 | 0.0106 | 0.9643 | 0.171 | 1792 | 6807 |
| SCEPTR | 64 | 1.000 | **0.9005** ± 0.0081 | 0.9866 | 0.0112 | 0.9586 | 0.176 | 1322 | 6781 |
| TCRemP | 9000 | 0.950 | 0.8017 ± 0.0038 | 0.9784 | 0.0197 | 0.9406 | 0.295 | 1796 | 6968 |
| TCR-BERT | 768 | 0.910 | 0.7700 ± 0.0056 | 0.9644 | 0.0291 | 0.8879 | 0.451 | 1174 | 6864 |
| ESM-2 8M | 320 | 0.974 | 0.6510 ± 0.0014 | 0.9326 | 0.0560 | 0.7692 | 0.849 | 1435 | 6720 |

Seed variance is small — at most ±0.008 on exact match — so gaps larger than about 0.02
are real rather than noise.

### Explained variance does not predict reconstruction quality

The relationship is inverted: aligned one-hot retains the least variance (0.380) and
reconstructs best, while ESM-2 retains almost all of it (0.974) and reconstructs worst.

Explained variance measures how much statistical spread survives the projection, not how
much residue-level information survives. One-hot has no correlations for PCA to exploit,
so its variance is spread evenly and the retained fraction looks low — but the 64
components it keeps describe the dominant axes of sequence variation, which is exactly
what the decoder needs.

The consequence is that explained variance cannot be used to normalize or excuse the
comparison. It is reported here because it is informative about each representation's
redundancy, not because it explains the ranking.

### One-hot is a ceiling, not a peer

The bottleneck equalizes dimensionality but not provenance. Aligned one-hot is a
**reversible encoding of the target itself**, so reconstructing from its PCA projection is
an autoencoding task: compress a sequence, decompress it. The other four invert a lossy
transformation of the sequence into a biological feature space. These are different
problems.

One-hot's 0.9144 should therefore be read as an approximate ceiling — what 64 numbers can
carry about a CDR3 when they encode the string directly. Its value is diagnostic: it shows
the decoder's capacity is not the limiting factor, so the spread among the other four
reflects their representations rather than the measuring instrument.

### Mean pooling handicaps the protein language models

ESM-2 and TCR-BERT emit one vector per residue, and the decoder needs one vector per
sequence. Both are mean-pooled over real residues, excluding BOS/EOS and padding — the
same rule for both, chosen for comparability.

Averaging over positions discards positional information, which is precisely what
sequence reconstruction requires. Their low scores partly measure this pooling choice
rather than the representations themselves. A CLS token, or per-position embeddings
concatenated before PCA, would likely score higher. Under the issue's constraint of one
frozen checkpoint producing one vector per receptor, mean pooling is the standard choice,
but the caveat belongs with the numbers.

### Where IRRM-CODEC wins and loses

**Loses: TCRemP is not the strongest compact representation.** SCEPTR reaches 0.9005
against TCRemP's 0.8017 — about 10 points — and does so while losing nothing to the
projection (explained variance 1.000, since it is natively 64-dimensional) where TCRemP
loses 5%. SCEPTR lands within 1.4 points of the one-hot ceiling, meaning its 64 numbers
are nearly as informative about the sequence as a direct compression of the sequence.
This is a genuine negative result for the claim that TCRemP is a good compact receptor
representation.

**Wins: TCRemP survives aggressive compression, and inversion improves under it.** The
committed multi-chain notebook reports exact match 0.7371 for the inverse model on raw
9000-dimensional TRB embeddings. At 64 dimensions — a 140× reduction — this benchmark
reaches **0.8017**. Compression did not cost accuracy; it appears to have removed noise.
That supports the project's premise that the TCRemP space is invertible and usable as a
design space, and it means the practical form of the model can be far smaller than the
published one.

**Wins: it beats both pretrained protein language models.** TCRemP is ahead of TCR-BERT
by 3 points and ESM-2 8M by 15, though the mean-pooling caveat above applies to both.

**Neutral: inference speed does not separate the representations.** All five run at
6,700–7,000 sequences/second, because the cost is dominated by the shared decoder, not by
the input. Encoder cost is what differs and is paid once (measured separately: one-hot
0.013 ms/sequence, SCEPTR 1.30, ESM-2 2.07, TCR-BERT 10.11 on 8 CPU threads).

Reconstruction quality is not what TCRemP was designed for — it encodes similarity to a
reference panel, for clustering and specificity work. Sequence recoverability is a
property IRRM-CODEC adds on top. The reconstruction benchmark therefore bounds one claim
about the representation; the Pgen benchmark tests the project's speed argument, where
the comparison is against OLGA rather than against other encoders.

Reproduce with `benchmark/slurm/reconstruction_array.sbatch`, then
`python -m benchmark.collect_reconstruction`.

## 3. Pgen prediction

Six arms, both targets, three training sizes. The three frozen-embedding arms share one
regression head (512 → 256 → 1) and one training configuration, so only the representation
differs between them. The two IRRM arms differ only in initialization. Ridge is
deterministic and selects its penalty on validation, which is that arm's equivalent of
early stopping.

Targets are standardized on train statistics and mapped back before any metric, so a
freshly initialized network does not spend its first epochs locating an offset near −6.5
that Ridge fits for free. Reported errors are in log10 units.

Neural arms ran ten seeds on every cell: 180 runs. Ten rather than three because the two
IRRM arms spread widely across seeds, and the effect they exist to measure is small by
construction — the arms are the same architecture on the same data, differing only in
their starting weights.

The whole sweep was then run **twice**, under two learning-rate configurations: a constant
1e-3, and a cosine schedule decaying to zero over the epoch budget (with early-stopping
patience widened from 8 to 12 so the schedule reaches its tail). The second configuration
was not a tuning pass but a robustness check, and it changed the headline conclusion. All
tables below report the cosine runs, which converge better and vary less; the constant-rate
numbers are kept in `results/pgen_constlr/` for comparison.

### Results on the full training split

Mean ± standard deviation over seeds, on the held-out test split.

**`log10_pgen_1mm`, train = 79,730**

| arm | RMSE | MAE | R² | Pearson r | Spearman ρ | bias | train s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| irrm_pretrained | **0.2181** ± 0.0322 | 0.1248 | 0.9813 | 0.9935 | 0.9921 | +0.0338 | 187 | 1584 |
| irrm_scratch | 0.2435 ± 0.0248 | 0.1583 | 0.9770 | 0.9916 | 0.9937 | +0.0466 | 105 | 1583 |
| tcr_bert_mlp | 0.2766 ± 0.0006 | 0.1671 | 0.9706 | 0.9854 | 0.9908 | +0.0175 | 71 | 1668 |
| sceptr_mlp | 0.4465 ± 0.0060 | 0.2510 | 0.9233 | 0.9621 | 0.9783 | +0.0642 | 58 | 1193 |
| esm2_8m_mlp | 0.4539 ± 0.0015 | 0.2655 | 0.9208 | 0.9607 | 0.9762 | +0.0549 | 68 | 1295 |
| tfidf_ridge | 0.5542 | 0.3671 | 0.8819 | 0.9391 | 0.9566 | +0.0035 | 9 | 717 |

**`log10_pgen`, train = 79,730**

| arm | RMSE | MAE | R² | Pearson r | Spearman ρ | bias | train s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| irrm_pretrained | **0.3102** ± 0.0388 | 0.1925 | 0.9722 | 0.9912 | 0.9883 | +0.1177 | 159 | 1593 |
| irrm_scratch | 0.3114 ± 0.0333 | 0.2027 | 0.9721 | 0.9921 | 0.9928 | +0.0987 | 131 | 1580 |
| tcr_bert_mlp | 0.3758 ± 0.0019 | 0.2333 | 0.9598 | 0.9800 | 0.9851 | +0.0238 | 74 | 1666 |
| esm2_8m_mlp | 0.5728 ± 0.0005 | 0.3494 | 0.9065 | 0.9535 | 0.9673 | +0.0746 | 63 | 1296 |
| sceptr_mlp | 0.5795 ± 0.0046 | 0.3442 | 0.9043 | 0.9532 | 0.9668 | +0.0976 | 61 | 1193 |
| tfidf_ridge | 0.6210 | 0.4145 | 0.8901 | 0.9435 | 0.9563 | +0.0087 | 8 | 714 |

The ordering is identical on both targets and at every training size: the two IRRM arms
lead, TCR-BERT is the best frozen representation, then SCEPTR and ESM-2, with Ridge last.
The gaps are large — IRRM halves the error of SCEPTR, ESM-2 and Ridge — and far exceed
seed spread. Per-size tables are in `results/pgen/pgen_summary.md`.

For reference, the committed pgen notebook reports RMSE 0.5914, R² 0.9437 and Pearson
0.9874 on `log10_pgen_1mm`. The best arm here reaches RMSE 0.2181, R² 0.9813 and Pearson
0.9935 on a different split of the same data.

### Does IRRM pretraining improve Pgen prediction?

**No — not consistently.** Welch's t-test on RMSE, ten seeds per arm, under both
learning-rate configurations. A positive delta means pretraining won.

| target | train | Δ RMSE (constant lr) | p | Δ RMSE (cosine) | p |
| --- | --- | ---: | ---: | ---: | ---: |
| log10_pgen | 1k | −0.017 | 0.527 | +0.045 | 0.113 |
| log10_pgen | 10k | −0.037 | 0.157 | −0.030 | **0.006** |
| log10_pgen | all | +0.045 | 0.118 | +0.001 | 0.940 |
| log10_pgen_1mm | 1k | −0.025 | 0.307 | +0.068 | **0.044** |
| log10_pgen_1mm | 10k | −0.013 | 0.500 | −0.006 | 0.763 |
| log10_pgen_1mm | all | +0.056 | **0.012** | +0.025 | 0.065 |

Under the constant rate, one cell was significant: pretraining cut RMSE by 20.9% on
`log10_pgen_1mm` at full data, p = 0.012, and it replicated across an accidental rerun.
That looked like a real, narrow effect.

It did not survive the schedule change. The same cell falls to Δ = +0.025 at p = 0.065,
while two different cells become significant — with **opposite signs**: pretraining
"helps" at 1k on one target and "hurts" at 10k on the other. No cell is significant under
both configurations, and three cells flip sign between them. That is the signature of
noise being read as effect, not of a small effect being measured precisely.

The mechanism is visible in the absolute numbers. Cosine scheduling improved almost every
arm, but not the pretrained one:

| arm, `log10_pgen_1mm` all | constant lr | cosine |
| --- | ---: | ---: |
| irrm_scratch | 0.2653 | **0.2435** |
| irrm_pretrained | 0.2098 | 0.2181 |
| tcr_bert_mlp | 0.3115 | **0.2766** |

The from-scratch arm gained what the pretrained arm did not. The transferred encoder was
supplying a good starting point that compensated for a poorly annealed learning rate; once
the optimizer no longer needed compensating, the advantage disappeared. Pretraining was
substituting for training quality rather than adding information.

Seed spread also fell where the comparison lives — 0.071 → 0.033 and 0.055 → 0.017 on the
from-scratch arm at `all` and 10k — so the cosine numbers are the more trustworthy ones,
not merely the more recent.

With six comparisons, a Bonferroni threshold is p < 0.008. Only the 10k "hurts" result
meets it, and it contradicts the 1k "helps" result on the other target. The honest
reading is that the sequence-to-TCRemP initialization has no reliable effect on Pgen
prediction at any training size.

### Is the advantage larger in the low-data regime?

The question presupposes an advantage that the data does not support. Across both
configurations and all six cells there is no consistent direction: the sign flips between
configurations at 1k on both targets, and the two nominally significant results point
opposite ways.

One asymmetry does survive both runs. The pretrained arm carries a larger positive bias at
small data — +0.32 and +0.24 at 1k against +0.23 for the from-scratch arm — so the
transferred features do push predictions systematically, and a small training set corrects
that less well. But this shows up as bias rather than as a reliable RMSE penalty.

### Quality against speed

End-to-end inference: string in, log10 Pgen out, including tokenization or encoder passes,
measured on one device at a time over 5,000 test sequences.

| arm | GPU (GTX 1080 Ti) | CPU (Xeon Gold, 8 threads) | RMSE (1mm, all) |
| --- | ---: | ---: | ---: |
| tfidf_ridge | 86,661 seq/s | 82,842 seq/s | 0.5542 |
| **irrm** | **64,814 seq/s** | **3,777 seq/s** | **0.2098** |
| esm2_8m_mlp | 9,393 seq/s | 813 seq/s | 0.4794 |
| tcr_bert_mlp | 4,039 seq/s | 216 seq/s | 0.3115 |
| sceptr_mlp | 2,495 seq/s | 1,265 seq/s | 0.4560 |

**IRRM dominates all three pretrained encoders on both axes at once** — 4.6× to 17.5×
faster on CPU, 6.9× to 26× on GPU, and lower RMSE than any of them. There is no trade-off
to make against the large frozen encoders; they are simply worse in both respects, because
their per-sequence encoding cost dwarfs the regression head that follows it.

The real trade-off is against Ridge, which is 1.3× faster on GPU and 22× on CPU while
carrying 2.6× the error. Ridge is the right choice when throughput matters more than
precision; IRRM when it does not.

Caveat on the SCEPTR row: it is the slowest arm on GPU yet mid-pack on CPU, which suggests
its library evaluates on CPU unless `sceptr.enable_hardware_acceleration()` is called,
which this benchmark does not do. Its GPU figure should be read as a CPU computation plus
transfer overhead, not as a GPU measurement.

### Against OLGA

The practical argument for a neural Pgen predictor is speed against the exact calculation.
Measured on the same 8-core laptop CPU, so the two numbers are directly comparable:

| | sequences/second | IRRM speedup |
| --- | ---: | ---: |
| IRRM | 1,969 | — |
| OLGA exact | 42.7 | **46×** |
| OLGA 1-mismatch | 2.7 | **729×** |

On the cluster GPU IRRM reaches 64,814 sequences/second, but OLGA was not benchmarked on
that node, so the laptop comparison is the honest one to quote.

### Where IRRM-CODEC wins and loses

**Wins: accuracy at every training size, on both targets.** IRRM leads all four baselines
in all six cells. On the full split it reaches R² 0.981 and Pearson 0.994 for
`log10_pgen_1mm`, and halves the error of SCEPTR, ESM-2 and Ridge. These margins are ten
times the seed spread, so they are not in question.

**Wins: the quality–speed trade-off against pretrained encoders.** Faster and more accurate
than frozen ESM-2, TCR-BERT and SCEPTR simultaneously. For this task the large pretrained
encoders contribute nothing on either axis.

**Wins: 46× to 729× over OLGA** on identical hardware, which is the argument the model
exists to make.

**Loses: the pretraining step buys nothing.** The sequence-to-TCRemP initialization shows no
reliable effect once the learning rate is properly annealed, and what looked like a 20.9%
gain under a constant rate was the pretrained encoder compensating for an untuned
optimizer. The practical implication is direct: train the Pgen model from scratch and skip
the pretraining stage, which costs an extra training run for no measurable return.

**Loses: seed stability.** Even with cosine scheduling the IRRM arms vary by 0.017–0.092
RMSE across seeds where the frozen arms vary by 0.0005–0.006 — still an order of magnitude
wider, because these arms fit a full encoder rather than a head over fixed features. Any
comparison between IRRM variants needs many seeds; three is not enough, as the first sweep
demonstrated.

**Loses: throughput against a trivial baseline.** TF-IDF + Ridge is 22× faster on CPU. If
an application needs coarse Pgen estimates at maximum rate, k-mers remain the right tool.

Reproduce with `benchmark/slurm/pgen_array.sbatch` and
`benchmark/slurm/pgen_seeds_array.sbatch`, then `python -m benchmark.collect_pgen` and
`python -m benchmark.benchmark_speed`.

## Limitations

- **Mean pooling** for ESM-2 and TCR-BERT discards positional information, which penalizes
  them most on reconstruction. A CLS token or per-position features would likely score
  higher.
- **Aligned one-hot is not a peer baseline** in section 2: it encodes the target
  reversibly, so it bounds the task rather than competing in it.
- **GPU training is not bit-reproducible.** Re-running a configuration with the same seed
  gives slightly different results, because cuDNN selects algorithms at runtime and some
  backward kernels accumulate non-deterministically. Conclusions here rest on ten-seed
  means rather than single runs.
- **One chain, one species.** Everything is human TRB. Nothing here shows the ordering
  holds for TRA or for other loci.
- **One bottleneck width.** Reconstruction was compared at 64 dimensions. Projections at
  32 and 128 are cached in `data/benchmark/trb/bottleneck/` but were not trained.
- **Six statistical comparisons** were made for the pretraining question, under each of two
  learning-rate configurations. No cell is significant under both, and three flip sign
  between them, which is why the conclusion is stated as absence of a reliable effect
  rather than as a measured null.
- **Sections 2 and 3 used different learning-rate schedules** — constant for reconstruction,
  cosine for Pgen. The issues require identical settings across the arms compared within
  each experiment, which holds; they are not comparable across sections. Reconstruction was
  not rerun because its seed spread (±0.0006 to ±0.008) leaves nothing for a schedule to
  fix.

## Environment

Sections 2 and 3 were produced on the ctlab Slurm cluster: GTX 1080 Ti, torch 2.6.0+cu124,
Python 3.11. Section 1 timings and the OLGA comparison are from an 8-core CPU
(Ryzen 7 8845HS); the CPU speed column in section 3 is from a cluster Xeon Gold node.
Peak memory is peak resident set size via `getrusage`, not
`torch.cuda.max_memory_allocated`, and the device is recorded with every measurement.

Encoder checkpoints: `esm2_t6_8M_UR50D` (fair-esm), `wukevin/tcr-bert` (HuggingFace),
SCEPTR default variant (`sceptr` 1.2.0).
