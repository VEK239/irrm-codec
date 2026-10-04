"""Shared-bottleneck Transformer for the joint RTP-CODEC objectives.

The model maps an amino-acid sequence to one compact latent vector and exposes
three task heads:

1. TCRemP embedding distillation;
2. log10(pgen) regression;
3. autoregressive amino-acid sequence reconstruction.

The heads return values in the standardized target spaces used during training.
The corresponding train-set means and standard deviations belong in the saved
checkpoint metadata and must be applied when reporting raw-space metrics.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from rtp_codec.tokenization.character import BOS_ID, EOS_ID, PAD_ID


@dataclass(frozen=True)
class RTPCodecConfig:
    input_vocab_size: int = 25
    output_vocab_size: int = 25
    share_input_output_embeddings: bool = True
    max_sequence_len: int = 40
    d_model: int = 320
    latent_dim: int = 128
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    ff_dim: int = 1280
    dropout: float = 0.1
    tcremp_dim: int = 9000
    tcremp_head_dim: int = 1024
    pgen_head_dim: int = 256
    decoder_memory_tokens: int = 4

    def validate(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.max_sequence_len < 1:
            raise ValueError("max_sequence_len must be positive.")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if self.decoder_memory_tokens < 1:
            raise ValueError("decoder_memory_tokens must be positive.")
        if self.input_vocab_size < 1 or self.output_vocab_size < 1:
            raise ValueError("Input and output vocabularies must be non-empty.")
        if (
            self.share_input_output_embeddings
            and self.input_vocab_size != self.output_vocab_size
        ):
            raise ValueError(
                "Input and output vocabulary sizes must match when embeddings are shared."
            )


class SequenceTransformerEncoder(nn.Module):
    """Encode a padded amino-acid token sequence into one latent vector."""

    def __init__(self, config: RTPCodecConfig, token_embedding: nn.Embedding):
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_sequence_len + 1, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.latent_projection = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens [batch, sequence], got {tuple(tokens.shape)}.")
        if tokens.size(1) > self.config.max_sequence_len:
            raise ValueError(
                f"Sequence width {tokens.size(1)} exceeds max_sequence_len="
                f"{self.config.max_sequence_len}."
            )

        if mask is None:
            mask = tokens.ne(PAD_ID)
        else:
            mask = mask.bool()
        if mask.shape != tokens.shape:
            raise ValueError("mask must have the same [batch, sequence] shape as tokens.")
        if not mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one non-padding token.")

        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, self.token_embedding(tokens)], dim=1)
        x = self.dropout(x + self.position_embedding[:, : x.size(1)])

        cls_is_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        valid_mask = torch.cat([cls_is_valid, mask], dim=1)
        encoded = self.transformer(x, src_key_padding_mask=~valid_mask)
        return self.latent_projection(encoded[:, 0])


class TCRemPHead(nn.Module):
    def __init__(self, config: RTPCodecConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.tcremp_head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tcremp_head_dim, config.tcremp_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class PgenHead(nn.Module):
    def __init__(self, config: RTPCodecConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.pgen_head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pgen_head_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).squeeze(-1)


class SequenceReconstructionHead(nn.Module):
    """Autoregressive decoder conditioned only on the compact latent vector."""

    def __init__(self, config: RTPCodecConfig, token_embedding: nn.Embedding):
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_sequence_len + 1, config.d_model)
        )
        self.memory_projection = nn.Linear(
            config.latent_dim,
            config.decoder_memory_tokens * config.d_model,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(
            config.d_model,
            config.output_vocab_size,
            bias=False,
        )
        self.output_projection.weight = token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _memory(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        return self.memory_projection(latent).reshape(
            batch_size,
            self.config.decoder_memory_tokens,
            self.config.d_model,
        )

    def forward(self, latent: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        if decoder_input.ndim != 2:
            raise ValueError("decoder_input must have shape [batch, sequence].")
        if decoder_input.size(0) != latent.size(0):
            raise ValueError("latent and decoder_input batch sizes do not match.")
        if decoder_input.size(1) > self.config.max_sequence_len + 1:
            raise ValueError("decoder_input is longer than the configured reconstruction limit.")

        x = self.token_embedding(decoder_input)
        x = self.dropout(x + self.position_embedding[:, : x.size(1)])
        causal_mask = torch.triu(
            torch.ones(
                decoder_input.size(1),
                decoder_input.size(1),
                dtype=torch.bool,
                device=decoder_input.device,
            ),
            diagonal=1,
        )
        decoded = self.transformer(
            tgt=x,
            memory=self._memory(latent),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input.eq(PAD_ID),
        )
        return self.output_projection(self.output_norm(decoded))

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        max_new_tokens: int | None = None,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
    ) -> torch.Tensor:
        limit = self.config.max_sequence_len + 1
        if max_new_tokens is not None:
            limit = min(limit, max_new_tokens)

        generated = torch.full(
            (latent.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=latent.device,
        )
        finished = torch.zeros(latent.size(0), dtype=torch.bool, device=latent.device)
        for _ in range(limit):
            logits = self(latent, generated)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if finished.all():
                break
        return generated[:, 1:]


class RTPCodecTransformer(nn.Module):
    """One shared sequence encoder and three jointly trainable task heads."""

    def __init__(self, config: RTPCodecConfig | None = None):
        super().__init__()
        self.config = config or RTPCodecConfig()
        self.config.validate()
        self.encoder_token_embedding = nn.Embedding(
            self.config.input_vocab_size,
            self.config.d_model,
            padding_idx=PAD_ID,
        )
        if self.config.share_input_output_embeddings:
            self.decoder_token_embedding = self.encoder_token_embedding
        else:
            self.decoder_token_embedding = nn.Embedding(
                self.config.output_vocab_size,
                self.config.d_model,
                padding_idx=PAD_ID,
            )
        self.encoder = SequenceTransformerEncoder(
            self.config,
            self.encoder_token_embedding,
        )
        self.tcremp_head = TCRemPHead(self.config)
        self.pgen_head = PgenHead(self.config)
        self.reconstruction_head = SequenceReconstructionHead(
            self.config,
            self.decoder_token_embedding,
        )

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(tokens, mask)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        latent = self.encode(tokens, mask)
        reconstruction_logits = None
        if decoder_input is not None:
            reconstruction_logits = self.reconstruction_head(latent, decoder_input)
        return {
            "latent": latent,
            "tcremp_standardized": self.tcremp_head(latent),
            "pgen_standardized": self.pgen_head(latent),
            "reconstruction_logits": reconstruction_logits,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        latent = self.encode(tokens, mask)
        generated = self.reconstruction_head.generate(latent, max_new_tokens=max_new_tokens)
        if was_training:
            self.train()
        return generated
