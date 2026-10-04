# Cluster execution

Install RTP-CODEC into the Python environment used by compute jobs. Submit from the repository root; scheduler partitions, GPU resources and memory limits must match the destination cluster.

The launchers are grouped under `research/scripts/slurm/{data,training,evaluation,preflight,reports,legacy}`. They preserve earlier experiment conditions and some external asset paths; review those paths before submitting. No jobs are launched merely by installing the package.

The compact manuscript launcher accepts any of the seven paper conditions:

```bash
export RTP_CODEC_ROOT="$PWD"
export RTP_CODEC_PYTHON="$PWD/.venv/bin/python"
export DATA_DIR=/path/to/locked/trb
export TOKENIZER_PATH=/path/to/data_anchor/anchored_tokenizer.json
sbatch research/scripts/slurm/training/paper.sbatch rtp
```

Change scheduler resource flags with `sbatch --partition=... --gres=...` as appropriate. Logs use the submission directory, and the preset writes results under `artifacts/paper/<condition>-seed42` unless `OUTPUT_DIR` is supplied.

Historical launchers may require preflight manifests, external encoder weights, TCRemP software and existing experiment directories. Their sequence/data gates should be satisfied before full jobs are submitted. Use the [reproduction guide](reproduction.md) and the corresponding [protocol](protocols) for asset requirements.
