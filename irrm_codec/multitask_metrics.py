"""Streaming metrics for joint IRRM-CODEC experiments."""

import math

import numpy as np
import torch

from irrm_codec.tokenization import AA_VOCAB, EOS_ID, PAD_ID


def _trim_token_row(tokens: list[int]) -> list[int]:
    result = []
    for token in tokens:
        if token == EOS_ID:
            break
        if token != PAD_ID:
            result.append(token)
    return result


def levenshtein_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including correct handling of tied values."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


class MultiTaskMetricAccumulator:
    def __init__(self):
        self.samples = 0
        self.loss_weight = 0
        self.loss_sums: dict[str, float] = {}

        self.tcremp_squared_error = 0.0
        self.tcremp_elements = 0
        self.tcremp_cosine_sum = 0.0

        self.pgen_squared_error = 0.0
        self.pgen_absolute_error = 0.0
        self.pgen_error = 0.0
        self.pgen_sum_pred = 0.0
        self.pgen_sum_target = 0.0
        self.pgen_sum_pred_sq = 0.0
        self.pgen_sum_target_sq = 0.0
        self.pgen_sum_cross = 0.0
        self.pgen_predictions: list[float] = []
        self.pgen_targets: list[float] = []

        self.correct_tokens = 0
        self.valid_tokens = 0
        self.generated_samples = 0
        self.exact_matches = 0
        self.length_matches = 0
        self.edit_distance_sum = 0.0
        self.generated_tokens = 0
        self.generated_valid_amino_acids = 0

    @torch.no_grad()
    def update(
        self,
        *,
        outputs: dict[str, torch.Tensor | None],
        losses: dict[str, torch.Tensor],
        tcremp_target: torch.Tensor,
        pgen_target: torch.Tensor,
        reconstruction_target: torch.Tensor,
        standardizer: dict[str, torch.Tensor],
        generated_tokens: torch.Tensor | None = None,
    ) -> None:
        batch_size = int(tcremp_target.size(0))
        self.samples += batch_size
        self.loss_weight += batch_size
        for name, value in losses.items():
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + float(value) * batch_size

        pred_tcremp_std = outputs["tcremp_standardized"]
        pred_pgen_std = outputs["pgen_standardized"]
        logits = outputs["reconstruction_logits"]
        if pred_tcremp_std is None or pred_pgen_std is None or logits is None:
            raise ValueError("All outputs are required to accumulate multitask metrics.")

        pred_tcremp = (
            pred_tcremp_std * standardizer["tcremp_std"]
            + standardizer["tcremp_mean"]
        )
        tcremp_diff = pred_tcremp - tcremp_target
        self.tcremp_squared_error += float(tcremp_diff.square().sum())
        self.tcremp_elements += tcremp_diff.numel()
        cosine = torch.nn.functional.cosine_similarity(
            pred_tcremp,
            tcremp_target,
            dim=-1,
        )
        self.tcremp_cosine_sum += float(cosine.sum())

        pred_pgen = pred_pgen_std * standardizer["pgen_std"] + standardizer["pgen_mean"]
        pgen_diff = pred_pgen - pgen_target
        self.pgen_squared_error += float(pgen_diff.square().sum())
        self.pgen_absolute_error += float(pgen_diff.abs().sum())
        self.pgen_error += float(pgen_diff.sum())
        self.pgen_sum_pred += float(pred_pgen.sum())
        self.pgen_sum_target += float(pgen_target.sum())
        self.pgen_sum_pred_sq += float(pred_pgen.square().sum())
        self.pgen_sum_target_sq += float(pgen_target.square().sum())
        self.pgen_sum_cross += float((pred_pgen * pgen_target).sum())
        self.pgen_predictions.extend(pred_pgen.detach().cpu().tolist())
        self.pgen_targets.extend(pgen_target.detach().cpu().tolist())

        token_predictions = logits.argmax(dim=-1)
        valid = reconstruction_target.ne(PAD_ID)
        self.correct_tokens += int(token_predictions.eq(reconstruction_target).logical_and(valid).sum())
        self.valid_tokens += int(valid.sum())

        if generated_tokens is not None:
            for predicted, target in zip(
                generated_tokens.detach().cpu().tolist(),
                reconstruction_target.detach().cpu().tolist(),
            ):
                predicted_trimmed = _trim_token_row(predicted)
                target_trimmed = _trim_token_row(target)
                self.generated_samples += 1
                self.exact_matches += int(predicted_trimmed == target_trimmed)
                self.length_matches += int(len(predicted_trimmed) == len(target_trimmed))
                self.edit_distance_sum += levenshtein_distance(
                    predicted_trimmed,
                    target_trimmed,
                )
                self.generated_tokens += len(predicted_trimmed)
                self.generated_valid_amino_acids += sum(
                    token > AA_VOCAB["-"] for token in predicted_trimmed
                )

    def compute(self, include_spearman: bool = True) -> dict[str, float | int]:
        if self.samples == 0:
            return {}
        metrics: dict[str, float | int] = {
            name: total / max(self.loss_weight, 1)
            for name, total in self.loss_sums.items()
        }
        n = self.samples
        pgen_rmse = math.sqrt(self.pgen_squared_error / n)
        target_total_variance = self.pgen_sum_target_sq - self.pgen_sum_target**2 / n
        r2 = (
            1.0 - self.pgen_squared_error / target_total_variance
            if target_total_variance > 0
            else float("nan")
        )
        pred_var = self.pgen_sum_pred_sq - self.pgen_sum_pred**2 / n
        target_var = target_total_variance
        covariance = self.pgen_sum_cross - self.pgen_sum_pred * self.pgen_sum_target / n
        pearson_denom = math.sqrt(max(pred_var, 0.0) * max(target_var, 0.0))
        pearson = covariance / pearson_denom if pearson_denom > 0 else float("nan")

        metrics.update(
            {
                "samples": n,
                "tcremp_mse_raw": self.tcremp_squared_error
                / max(self.tcremp_elements, 1),
                "tcremp_cosine_raw": self.tcremp_cosine_sum / n,
                "pgen_rmse_raw": pgen_rmse,
                "pgen_mae_raw": self.pgen_absolute_error / n,
                "pgen_bias_raw": self.pgen_error / n,
                "pgen_r2_raw": r2,
                "pgen_pearson_raw": pearson,
                "reconstruction_token_accuracy_teacher_forced": self.correct_tokens
                / max(self.valid_tokens, 1),
            }
        )
        if include_spearman:
            pgen_pred = np.asarray(self.pgen_predictions, dtype=np.float64)
            pgen_target = np.asarray(self.pgen_targets, dtype=np.float64)
            metrics["pgen_spearman_raw"] = _pearson(
                _rankdata(pgen_pred),
                _rankdata(pgen_target),
            )
        if self.generated_samples:
            metrics.update(
                {
                    "reconstruction_exact_match": self.exact_matches
                    / self.generated_samples,
                    "reconstruction_length_accuracy": self.length_matches
                    / self.generated_samples,
                    "reconstruction_mean_edit_distance": self.edit_distance_sum
                    / self.generated_samples,
                    "generated_samples": self.generated_samples,
                    "reconstruction_valid_aa_fraction": self.generated_valid_amino_acids
                    / max(self.generated_tokens, 1),
                }
            )
        return metrics
