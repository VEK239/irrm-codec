# Leakage-safe sequence-only anchored tokenizer study

## Formal mapping

For a validated amino-acid CDR3 string `s`, each candidate implements

`T(s) = [a_N(s), WP(m(s)), a_C(s)]`,

where `a_N(s)` and `a_C(s)` are literal substrings of `s`, `m(s)` is exactly
the intervening substring, and `WP` is a WordPiece model fitted only to middle
spans from a benchmark-disjoint independently generated TRB corpus. Concatenating
the decoded token surfaces must reproduce `s` byte-for-byte. The encoder accepts
only `s`; neither `v_call` nor `j_call` is an inference input. The reconstruction
decoder remains the unchanged 25-token character decoder.

All tokenizers reserve PAD/BOS/EOS/UNK IDs 0/1/2/3. Normal validated inputs never
emit UNK. Input tokens have exactly one of three boundary types: `N_ANCHOR`,
`MIDDLE_WORDPIECE`, or `C_ANCHOR`.

## Candidates

1. **EDGE-3 (`edge_k`)** protects the first and last three residues as one token
   each. Its vocabulary exhaustively enumerates every 3-mer over the 20
   standard amino acids (8,000 tokens per side). This makes every valid literal
   edge representable without observing evaluation sequences. K=3 is
   the largest feasible non-overlapping choice because the locked benchmark's
   recorded minimum CDR3 length is seven; K=4 would require eight residues and
   failed the CPU coverage gate. Inputs shorter than six or containing invalid
   amino acids fail closed.
2. **DATA-ANCHOR (`data_anchor`)** mines literal prefixes and suffixes of length
   2–8 from locked train rows only. A motif is retained when it has at least 50
   train occurrences and at least 25 occurrences within one annotated V (N side)
   or J (C side) gene. V/J calls restrict the train-only library construction but
   are discarded before encoding. All 20 one-residue identity fallbacks are
   reserved constants (not learned motifs), guaranteeing sequence-only coverage
   without held-out fitting.
3. **GERMLINE-ANCHOR (`germline_anchor`)** translates the functional V and J
   CDR3-compatible segments in the documented OLGA human `TRB_orig` model using
   its V/J anchor-index CSVs. Literal prefixes/suffixes of length 1–8 are retained
   only if at least 20 locked train sequences support them. The exact source-file
   hashes are recorded. All 20 one-residue identity fallbacks are reserved
   constants; multi-residue germline motifs remain reference-derived and
   train-supported. Encoding still operates solely by literal sequence match.

For variable anchors, all matching prefix/suffix pairs that do not overlap are
considered. Selection maximizes total protected length, then N length, then C
length, then uses lexical order. Distinct same-length literal matches cannot both
match the same edge; collision counts and overlap-driven fallback counts are
nonetheless audited. If no identity fallback pair exists, encoding fails closed.

## Corpus and leakage controls

The preferred central corpus is a deterministic 500,000-row subset streamed from
the independently generated OLGA human TRB corpus
`/projects/immunestatus/vdjrearm/sample/TRB_1e7.tsv.gz`. Its generation script,
OLGA model files, and corpus are hashed. Every locked train, validation, and test
CDR3 is removed before fitting, duplicates and invalid rows are removed, and the
recorded final overlap must be zero. The external corpus influences central
WordPiece only; DATA and GERMLINE anchor retention uses locked train evidence
only. Validation and test strings are used solely for post-fit audits and metrics.

## Matched experiment rule

All three candidates use full R+T+P supervision, seed 42, d-model 320, latent 128,
four encoder and four decoder layers, batch 64, 40 epochs, AdamW at 3e-4, and the
locked train-only normalizer. Apart from input tokenizer path/type and the implied
input embedding vocabulary size/parameter count, configurations match the selected
character baseline. Selection uses validation metrics only; test metrics are
descriptive. Any annotation-aware V/J-input condition would be a separate modality
and is outside this study.
