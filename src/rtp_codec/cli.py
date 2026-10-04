"""Public command line entry points."""
import argparse
import json
import sys
from pathlib import Path

def training_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    args, remaining = parser.parse_known_args(argv)
    if args.config is None:
        return remaining
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    arguments = payload["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("Configuration 'arguments' must be an object.")
    prefix = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            prefix.append(flag if value else "--no-" + key.replace("_", "-"))
        elif value is not None:
            prefix.extend([flag, str(value)])
    return prefix + remaining

def train():
    from rtp_codec.training.multitask import main as train_main
    original = sys.argv
    try:
        sys.argv = [original[0], *training_arguments(original[1:])]
        train_main()
    finally:
        sys.argv = original

def main():
    parser = argparse.ArgumentParser(description="RTP-CODEC training and sequence encoding")
    parser.add_argument("command", choices=("train", "encode"))
    if len(sys.argv) == 1 or sys.argv[1] in ("--help", "-h"):
        parser.print_help()
        return
    if sys.argv[1] not in ("train", "encode"):
        parser.error(f"unknown command: {sys.argv[1]}")
    original = sys.argv
    try:
        sys.argv = [original[0], *original[2:]]
        if original[1] == "train":
            train()
        else:
            from rtp_codec.inference import main as encode_main
            encode_main()
    finally:
        sys.argv = original
