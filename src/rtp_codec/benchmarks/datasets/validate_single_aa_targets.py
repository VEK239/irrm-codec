"""Identity-align independently scored mutation-challenge targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def align_targets(prepared: Path, pgen_path: Path, representations_path: Path,
                  tcremp_path: Path, output_dir: Path) -> dict:
    mutants = pd.read_csv(prepared / "mutants.tsv", sep="\t", dtype={"clone_id": str})
    pgen = pd.read_csv(pgen_path, sep="\t", dtype={"clone_id": str})
    representations = pd.read_csv(representations_path, sep="\t", dtype={"clone_id": str})
    tcremp = pd.read_parquet(tcremp_path)
    for name, frame in (("mutants", mutants), ("pgen", pgen),
                        ("representations", representations), ("tcremp", tcremp)):
        if "clone_id" not in frame or frame["clone_id"].isna().any() or frame["clone_id"].duplicated().any():
            raise ValueError(f"{name} lacks complete unique clone_id values.")
    expected = mutants["clone_id"].astype(str).tolist()
    expected_set = set(expected)
    for name, frame in (("pgen", pgen), ("representations", representations), ("tcremp", tcremp)):
        observed = set(frame["clone_id"].astype(str))
        if observed != expected_set:
            raise ValueError(f"{name} clone IDs differ: missing={len(expected_set-observed)} extra={len(observed-expected_set)}")
    pgen = pgen.set_index("clone_id").loc[expected]
    representations = representations.set_index("clone_id").loc[expected]
    tcremp = tcremp.set_index("clone_id").loc[expected]
    sidecar_sequence_column = next(
        (column for column in ("cdr3aa_TRB", "junction_aa") if column in representations), None
    )
    if sidecar_sequence_column is None:
        raise ValueError(f"TCRemP sidecar lacks a recognized TRB sequence column: {list(representations.columns)}")
    if not np.array_equal(representations[sidecar_sequence_column].astype(str).to_numpy(), mutants["junction_aa"].astype(str).to_numpy()):
        raise ValueError("TCRemP sidecar sequences do not align to generated mutants by clone_id.")
    if not np.array_equal(pgen["junction_aa"].astype(str).to_numpy(), mutants["junction_aa"].astype(str).to_numpy()):
        raise ValueError("Pgen sequences do not align to generated mutants by clone_id.")
    exact_values = pd.to_numeric(pgen["log10_pgen"], errors="coerce").to_numpy(np.float64)
    target_values = pd.to_numeric(pgen["log10_pgen_1mm"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError(
            f"Nonfinite declared log10_pgen_1mm targets: {int((~np.isfinite(target_values)).sum())}"
        )
    matrix = tcremp.to_numpy(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 9000 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid TCRemP matrix {matrix.shape}.")
    output_dir.mkdir(parents=True, exist_ok=False)
    matrix_path = output_dir / "mutant_tcremp.npy"
    target_path = output_dir / "targets.tsv"
    np.save(matrix_path, matrix, allow_pickle=False)
    result = mutants.copy()
    # Exact pgen can legitimately be zero for a valid mutant. It is not an
    # evaluation target; retain its nonfinite count as a diagnostic but keep
    # only the finite 1-mismatch target in the accepted table.
    result["log10_pgen_1mm"] = target_values
    result.to_csv(target_path, sep="\t", index=False, lineterminator="\n")
    report = {
        "status": "accepted_for_frozen_delta_evaluation",
        "alignment": "unique clone_id plus exact junction_aa agreement",
        "tcremp_sidecar_sequence_column": sidecar_sequence_column,
        "rows": len(result),
        "tcremp_shape": list(matrix.shape),
        "pgen_target": "log10_pgen_1mm",
        "finite_pgen_target": True,
        "nonfinite_exact_log10_pgen_diagnostic_rows": int((~np.isfinite(exact_values)).sum()),
        "finite_tcremp": True,
        "sources": {str(path): sha256(path) for path in (pgen_path, representations_path, tcremp_path)},
        "artifacts": {matrix_path.name: sha256(matrix_path), target_path.name: sha256(target_path)},
    }
    (output_dir / "TARGETS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pgen", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--tcremp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(align_targets(args.prepared, args.pgen, args.representations,
                                   args.tcremp, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
