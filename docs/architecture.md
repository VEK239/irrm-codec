# Architecture

RTP-CODEC encodes a human TRB CDR3 amino-acid sequence into a shared latent vector. In the manuscript configuration, the encoder and decoder use hidden width 320, eight attention heads, four layers each, FFN width 1,280 and dropout 0.1. The bottleneck has 128 coordinates. The maximum CDR3 length is 40 residues.

The encoder prepends a learned CLS token, adds learned positional embeddings and projects the CLS state through normalization, a linear layer and GELU into the bottleneck. The reconstruction decoder attends to memory tokens projected from this vector and predicts character tokens autoregressively.

The DATA-ANCHOR tokenizer uses train-derived edge motifs and a separately trained middle WordPiece vocabulary. Encoder and decoder vocabularies may differ; reconstruction is always character-level. Tokenizer IDs and bundle files are part of checkpoint compatibility.

## Training objectives

- **R:** character cross-entropy, ignoring PAD targets.
- **T:** regression to standardized TCRemP coordinates with a mixture of MSE and cosine loss (MSE fraction 0.7). The head uses width 1,024 and predicts 9,000 coordinates in the reported benchmark.
- **P:** Huber regression (delta 0.5) to standardized `log10_pgen_1mm`, using a head of width 256.

Full RTP training assigns weight 1 to each term. Inactive objectives have weight 0. Normalization statistics are fitted only on the training rows. TCRemP and Pgen targets are precomputed; sequence encoding after training does not run either target generator.

Implementation: [codec](../src/rtp_codec/models/codec.py), [objectives](../src/rtp_codec/training/objectives.py), [trainer](../src/rtp_codec/training/multitask.py). The seven [paper presets](../research/configs/paper) hold architecture and optimization settings fixed while changing objective weights.

Earlier configurations under `research/configs/trb` include other bottleneck widths and tokenizer choices. Always load the architecture recorded in a checkpoint rather than substituting the current defaults.
