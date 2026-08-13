"""Train and evaluate the shared-bottleneck IRRM-CODEC Transformer."""

import argparse
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from irrm_codec.multitask_data import (
    MultiTaskBenchmarkDataset,
    TargetStandardizer,
    build_multitask_dataloader,
    compute_target_standardizer,
    load_prepared_benchmark,
    resolve_encoder_tokenizer,
    select_split_indices,
)
from irrm_codec.multitask_losses import IRRMCodecMultiTaskLoss, MultiTaskLossWeights
from irrm_codec.multitask_metrics import MultiTaskMetricAccumulator
from irrm_codec.multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer
from irrm_codec.tokenization import AA_VOCAB
from irrm_codec.utils import save_json, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/benchmark/trb")
    parser.add_argument("--output-dir", default="artifacts/multitask/char")
    parser.add_argument("--train-subset", choices=["1k", "10k", "all"], default="all")
    parser.add_argument(
        "--pgen-target",
        choices=["log10_pgen", "log10_pgen_1mm"],
        default="log10_pgen_1mm",
    )
    parser.add_argument("--tokenizer-type", choices=["char", "wordpiece"], default="char")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--max-sequence-len", type=int, default=40)

    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--latent-dim", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tcremp-head-dim", type=int, default=1024)
    parser.add_argument("--pgen-head-dim", type=int, default=256)
    parser.add_argument("--decoder-memory-tokens", type=int, default=4)

    parser.add_argument("--tcremp-loss-weight", type=float, default=1.0)
    parser.add_argument("--pgen-loss-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=1.0)
    parser.add_argument("--tcremp-mse-fraction", type=float, default=0.7)
    parser.add_argument("--pgen-huber-delta", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--standardizer-chunk-size", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-generation-metrics", action="store_true")
    parser.add_argument(
        "--save-test-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-batch latent vectors and raw-space test predictions for analysis.",
    )
    parser.add_argument(
        "--val-generation-every",
        type=int,
        default=0,
        help="Compute slow autoregressive validation metrics every N epochs; 0 disables them.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Debug limit; 0 uses the whole train loader.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Debug limit; 0 uses the whole validation/test loader.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "standardizer_chunk_size": args.standardizer_chunk_size,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    for name in (
        "tcremp_loss_weight",
        "pgen_loss_weight",
        "reconstruction_loss_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if (
        args.tcremp_loss_weight
        + args.pgen_loss_weight
        + args.reconstruction_loss_weight
        == 0
    ):
        raise ValueError("At least one task loss weight must be positive.")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _batch_limit(loader, requested: int) -> int:
    return min(len(loader), requested) if requested > 0 else len(loader)


def run_epoch(
    *,
    model: IRRMCodecTransformer,
    criterion: IRRMCodecMultiTaskLoss,
    loader,
    standardizer_tensors: dict[str, torch.Tensor],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    amp_enabled: bool,
    generate: bool,
    max_batches: int,
    log_interval: int,
    show_progress: bool,
    stage: str,
    prediction_dir: Path | None = None,
) -> dict[str, float | int]:
    is_train = optimizer is not None
    model.train(is_train)
    accumulator = MultiTaskMetricAccumulator()
    limit = _batch_limit(loader, max_batches)
    progress = tqdm(
        loader,
        total=limit,
        desc=stage,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    prediction_files = []
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(progress, start=1):
        if step > limit:
            break
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(
                    batch["encoder_tokens"],
                    batch["encoder_mask"],
                    batch["decoder_input"],
                )
                losses = criterion(
                    outputs,
                    tcremp_target=batch["tcremp_target"],
                    pgen_target=batch["pgen_target"],
                    reconstruction_target=batch["reconstruction_target"],
                    **standardizer_tensors,
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(
                        f"Non-finite {stage} loss at batch {step}: "
                        f"{float(losses['loss'].detach())}"
                    )

            if is_train:
                scaled_loss = losses["loss"] / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                should_step = (
                    step % gradient_accumulation_steps == 0 or step == limit
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        generated_tokens = None
        if generate:
            generated_tokens = model.reconstruction_head.generate(outputs["latent"])
        accumulator.update(
            outputs=outputs,
            losses=losses,
            tcremp_target=batch["tcremp_target"],
            pgen_target=batch["pgen_target"],
            reconstruction_target=batch["reconstruction_target"],
            standardizer=standardizer_tensors,
            generated_tokens=generated_tokens,
        )
        if prediction_dir is not None:
            pred_tcremp = (
                outputs["tcremp_standardized"] * standardizer_tensors["tcremp_std"]
                + standardizer_tensors["tcremp_mean"]
            )
            pred_pgen = (
                outputs["pgen_standardized"] * standardizer_tensors["pgen_std"]
                + standardizer_tensors["pgen_mean"]
            )
            prediction_path = prediction_dir / f"batch_{step:05d}.npz"
            arrays = {
                "row_index": batch["row_index"].detach().cpu().numpy(),
                "latent": outputs["latent"].detach().float().cpu().numpy(),
                "tcremp_prediction": pred_tcremp.detach().float().cpu().numpy(),
                "pgen_prediction": pred_pgen.detach().float().cpu().numpy(),
                "pgen_target": batch["pgen_target"].detach().float().cpu().numpy(),
                "reconstruction_target": batch["reconstruction_target"].detach().cpu().numpy(),
            }
            if generated_tokens is not None:
                arrays["generated_tokens"] = generated_tokens.detach().cpu().numpy()
            np.savez(prediction_path, **arrays)
            prediction_files.append(prediction_path.name)
        if show_progress and (step == limit or (log_interval > 0 and step % log_interval == 0)):
            interim = accumulator.compute(include_spearman=False)
            progress.set_postfix(
                loss=f"{interim['loss']:.4f}",
                cosine=f"{interim['tcremp_cosine_raw']:.4f}",
                pgen_rmse=f"{interim['pgen_rmse_raw']:.4f}",
                token_acc=f"{interim['reconstruction_token_accuracy_teacher_forced']:.4f}",
            )
    progress.close()
    metrics = accumulator.compute()
    if prediction_dir is not None:
        save_json(
            prediction_dir / "manifest.json",
            {
                "stage": stage,
                "samples": metrics.get("samples", 0),
                "batches": len(prediction_files),
                "files": prediction_files,
                "metrics": metrics,
            },
        )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: IRRMCodecTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    metrics: dict,
    best_val_loss: float,
    epochs_without_improvement: int,
    model_config: IRRMCodecConfig,
    tokenizer_info: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "model_config": asdict(model_config),
        "tokenizer": tokenizer_info,
        "training_args": vars(args),
        "standardizer_path": "target_standardizer.npz",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configs_match(expected: IRRMCodecConfig, payload: dict) -> bool:
    return asdict(expected) == payload


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "train.log")
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    logger.info("loading prepared benchmark from %s", Path(args.data_dir).resolve())
    table, embeddings = load_prepared_benchmark(args.data_dir)
    train_indices = select_split_indices(
        table,
        args.data_dir,
        "train",
        train_subset=args.train_subset,
    )
    val_indices = select_split_indices(table, args.data_dir, "val")
    test_indices = select_split_indices(table, args.data_dir, "test")
    tokenizer = resolve_encoder_tokenizer(args.tokenizer_type, args.tokenizer_path)

    standardizer_path = output_dir / "target_standardizer.npz"
    if args.resume and standardizer_path.exists():
        standardizer = TargetStandardizer.load(standardizer_path)
        if standardizer.pgen_target != args.pgen_target:
            raise ValueError("Resume checkpoint and requested pgen target do not match.")
        if standardizer.train_rows != len(train_indices):
            raise ValueError("Resume standardizer and requested train subset do not match.")
        if len(standardizer.tcremp_mean) != embeddings.shape[1]:
            raise ValueError("Resume standardizer and TCRemP dimension do not match.")
    else:
        logger.info(
            "computing train-only target statistics rows=%d embedding_dim=%d",
            len(train_indices),
            embeddings.shape[1],
        )
        standardizer = compute_target_standardizer(
            table,
            embeddings,
            train_indices,
            args.pgen_target,
            chunk_size=args.standardizer_chunk_size,
        )
        standardizer.save(standardizer_path)
    standardizer_tensors = standardizer.as_torch(device)

    datasets = {
        "train": MultiTaskBenchmarkDataset(
            table,
            embeddings,
            train_indices,
            tokenizer,
            args.pgen_target,
            args.max_sequence_len,
        ),
        "val": MultiTaskBenchmarkDataset(
            table,
            embeddings,
            val_indices,
            tokenizer,
            args.pgen_target,
            args.max_sequence_len,
        ),
        "test": MultiTaskBenchmarkDataset(
            table,
            embeddings,
            test_indices,
            tokenizer,
            args.pgen_target,
            args.max_sequence_len,
        ),
    }
    loaders = {
        split: build_multitask_dataloader(
            dataset,
            args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    model_config = IRRMCodecConfig(
        input_vocab_size=tokenizer.vocab_size,
        output_vocab_size=len(AA_VOCAB),
        share_input_output_embeddings=tokenizer.name == "char",
        max_sequence_len=args.max_sequence_len,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        tcremp_dim=int(embeddings.shape[1]),
        tcremp_head_dim=args.tcremp_head_dim,
        pgen_head_dim=args.pgen_head_dim,
        decoder_memory_tokens=args.decoder_memory_tokens,
    )
    model = IRRMCodecTransformer(model_config).to(device)
    criterion = IRRMCodecMultiTaskLoss(
        weights=MultiTaskLossWeights(
            tcremp=args.tcremp_loss_weight,
            pgen=args.pgen_loss_weight,
            reconstruction=args.reconstruction_loss_weight,
        ),
        tcremp_mse_fraction=args.tcremp_mse_fraction,
        pgen_huber_delta=args.pgen_huber_delta,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    tokenizer_info = {
        "type": tokenizer.name,
        "path": tokenizer.path,
        "input_vocab_size": tokenizer.vocab_size,
        "output_vocab_size": len(AA_VOCAB),
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_json(
        output_dir / "run_config.json",
        {
            "model": asdict(model_config),
            "tokenizer": tokenizer_info,
            "training": vars(args),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "parameter_count": parameter_count,
        },
    )
    save_json(
        output_dir / "data_stats.json",
        {
            "data_dir": str(Path(args.data_dir).resolve()),
            "train_subset": args.train_subset,
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
            "test_rows": len(test_indices),
            "tcremp_dim": int(embeddings.shape[1]),
            "pgen_target": args.pgen_target,
            "standardizer": str(standardizer_path),
        },
    )
    logger.info(
        "model ready tokenizer=%s input_vocab=%d output_vocab=%d parameters=%d "
        "train=%d val=%d test=%d device=%s amp=%s",
        tokenizer.name,
        tokenizer.vocab_size,
        len(AA_VOCAB),
        parameter_count,
        len(train_indices),
        len(val_indices),
        len(test_indices),
        device,
        amp_enabled,
    )

    start_epoch = 1
    best_val_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if not configs_match(model_config, checkpoint["model_config"]):
            raise ValueError("Resume checkpoint model configuration does not match this run.")
        if checkpoint.get("tokenizer") != tokenizer_info:
            raise ValueError("Resume checkpoint tokenizer configuration does not match this run.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        logger.info("resumed checkpoint=%s at epoch=%d", args.resume, start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=loaders["train"],
            standardizer_tensors=standardizer_tensors,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            amp_enabled=amp_enabled,
            generate=False,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval,
            show_progress=not args.no_progress,
            stage=f"train {epoch}/{args.epochs}",
            prediction_dir=None,
        )
        generate_val = (
            args.val_generation_every > 0
            and epoch % args.val_generation_every == 0
        )
        val_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=loaders["val"],
            standardizer_tensors=standardizer_tensors,
            device=device,
            optimizer=None,
            scaler=scaler,
            gradient_accumulation_steps=1,
            max_grad_norm=0,
            amp_enabled=amp_enabled,
            generate=generate_val,
            max_batches=args.max_eval_batches,
            log_interval=args.log_interval,
            show_progress=not args.no_progress,
            stage=f"val {epoch}/{args.epochs}",
            prediction_dir=None,
        )
        scheduler.step(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        save_json(output_dir / "history.json", history)

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = float(val_metrics["loss"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "model_config": model_config,
            "tokenizer_info": tokenizer_info,
            "args": args,
        }
        save_checkpoint(output_dir / "last.pt", **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / "best.pt", **checkpoint_kwargs)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_cosine=%.4f "
            "val_pgen_rmse=%.4f val_token_acc=%.4f best_val_loss=%.4f patience=%d/%d",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["tcremp_cosine_raw"],
            val_metrics["pgen_rmse_raw"],
            val_metrics["reconstruction_token_accuracy_teacher_forced"],
            best_val_loss,
            epochs_without_improvement,
            args.early_stopping_patience,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            logger.info("early stopping at epoch=%d", epoch)
            break

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = run_epoch(
        model=model,
        criterion=criterion,
        loader=loaders["test"],
        standardizer_tensors=standardizer_tensors,
        device=device,
        optimizer=None,
        scaler=scaler,
        gradient_accumulation_steps=1,
        max_grad_norm=0,
        amp_enabled=amp_enabled,
        generate=not args.skip_generation_metrics,
        max_batches=args.max_eval_batches,
        log_interval=args.log_interval,
        show_progress=not args.no_progress,
        stage="test",
        prediction_dir=(
            output_dir / "test_predictions" if args.save_test_predictions else None
        ),
    )
    test_metrics.update(
        {
            "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
            "best_checkpoint_val_loss": float(best_checkpoint["best_val_loss"]),
            "parameter_count": parameter_count,
            "tokenizer_type": tokenizer.name,
            "train_subset": args.train_subset,
        }
    )
    save_json(output_dir / "test_metrics.json", test_metrics)
    logger.info("test metrics: %s", json.dumps(test_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
