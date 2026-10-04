| Method | Modality | Reconstruction exact | pgen RMSE | VDJdb P@1 | P@5 | P@10 | MAP | Within/between |
|---|---|---|---|---|---|---|---|---|
| DATA-ANCHOR RTP | sequence-only | 0.8831 +/- 0.0058 | 0.2375 +/- 0.0008 | 0.3879 | 0.3272 | 0.2940 | 0.3136 | 0.9526 |
| TCRemP prototype distances | sequence-derived alignment | 0.5181 +/- 0.0024 | 0.6038 +/- 0.0019 | NA | NA | NA | NA | NA |
| TCR-BERT final residue mean | sequence-only | 0.4778 +/- 0.0179 | 0.4299 +/- 0.0022 | 0.3590 | 0.3053 | 0.2735 | 0.2999 | 0.9722 |
| ESM2-35M final residue mean | sequence-only | 0.2042 +/- 0.0132 | 0.7086 +/- 0.0033 | 0.3597 | 0.3081 | 0.2782 | 0.2968 | 0.9410 |
| ESM2-8M final residue mean | sequence-only sensitivity | 0.2179 +/- 0.0075 | 0.7439 +/- 0.0012 | NA | NA | NA | NA | NA |
| SCEPTR CDR3-only native CLS | sequence-only | 0.8213 +/- 0.0043 | 0.6637 +/- 0.0062 | 0.3553 | 0.2778 | 0.2366 | 0.2651 | 1.0469 |
| SCEPTR default native CLS | annotation-aware V+CDR3+J | 0.7263 +/- 0.0125 | 0.8073 +/- 0.0084 | NA | NA | NA | NA | NA |
| CDR3 TF-IDF ridge | sequence control | NA | 0.7860 | NA | NA | NA | NA | NA |
