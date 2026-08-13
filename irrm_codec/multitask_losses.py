"""Losses for joint IRRM-CODEC Transformer training."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from irrm_codec.tokenization import PAD_ID


@dataclass(frozen=True)
class MultiTaskLossWeights:
    tcremp: float = 1.0
    pgen: float = 1.0
    reconstruction: float = 1.0


class IRRMCodecMultiTaskLoss(nn.Module):
    """Combine normalized losses without mixing incompatible target scales.

    The TCRemP and pgen heads predict standardized train-target values. TCRemP
    cosine loss is nevertheless computed after de-standardization, so the model
    is optimized and evaluated in the original embedding geometry as well.
    """

    def __init__(
        self,
        weights: MultiTaskLossWeights | None = None,
        tcremp_mse_fraction: float = 0.7,
        pgen_huber_delta: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= tcremp_mse_fraction <= 1.0:
            raise ValueError("tcremp_mse_fraction must lie in [0, 1].")
        self.weights = weights or MultiTaskLossWeights()
        self.tcremp_mse_fraction = tcremp_mse_fraction
        self.pgen_huber_delta = pgen_huber_delta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        *,
        tcremp_target: torch.Tensor,
        pgen_target: torch.Tensor,
        reconstruction_target: torch.Tensor,
        tcremp_mean: torch.Tensor,
        tcremp_std: torch.Tensor,
        pgen_mean: torch.Tensor,
        pgen_std: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_tcremp_std = outputs["tcremp_standardized"]
        pred_pgen_std = outputs["pgen_standardized"]
        reconstruction_logits = outputs["reconstruction_logits"]
        if pred_tcremp_std is None or pred_pgen_std is None or reconstruction_logits is None:
            raise ValueError("All three task outputs are required for joint training.")

        safe_tcremp_std = tcremp_std.clamp_min(1e-8)
        target_tcremp_std = (tcremp_target - tcremp_mean) / safe_tcremp_std
        tcremp_mse = F.mse_loss(pred_tcremp_std, target_tcremp_std)
        pred_tcremp_raw = pred_tcremp_std * safe_tcremp_std + tcremp_mean
        tcremp_cosine = 1.0 - F.cosine_similarity(
            pred_tcremp_raw,
            tcremp_target,
            dim=-1,
        ).mean()
        tcremp_loss = (
            self.tcremp_mse_fraction * tcremp_mse
            + (1.0 - self.tcremp_mse_fraction) * tcremp_cosine
        )

        safe_pgen_std = pgen_std.clamp_min(1e-8)
        target_pgen_std = (pgen_target - pgen_mean) / safe_pgen_std
        pgen_loss = F.huber_loss(
            pred_pgen_std,
            target_pgen_std,
            delta=self.pgen_huber_delta,
        )

        reconstruction_loss = F.cross_entropy(
            reconstruction_logits.reshape(-1, reconstruction_logits.size(-1)),
            reconstruction_target.reshape(-1),
            ignore_index=PAD_ID,
            label_smoothing=self.label_smoothing,
        )

        total = (
            self.weights.tcremp * tcremp_loss
            + self.weights.pgen * pgen_loss
            + self.weights.reconstruction * reconstruction_loss
        )
        return {
            "loss": total,
            "tcremp_loss": tcremp_loss,
            "tcremp_mse_standardized": tcremp_mse,
            "tcremp_cosine_loss_raw": tcremp_cosine,
            "pgen_loss": pgen_loss,
            "reconstruction_loss": reconstruction_loss,
        }
