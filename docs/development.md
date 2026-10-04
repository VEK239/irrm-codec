# Development

Install from a checkout with `python -m pip install -e ".[dev]"`. Source lives under `src/rtp_codec`; tests live under `tests`. Keep reusable model/data logic in the library, experiment orchestration in `experiments` and task-specific comparisons in `benchmarks`.

Run before proposing a code change:

```bash
python -m compileall -q src tests
python -m pytest -q
rtp-codec --help
rtp-codec train --help
rtp-codec encode --help
find scripts -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 | xargs -0 -n1 bash -n
```

Add tests for changed scientific behavior, asset alignment and checkpoint compatibility. CPU tests use small synthetic fixtures; expensive experiments are validated through their documented preflight checks and saved result provenance.

Never commit full datasets, model weights, environments or generated run directories. Record configuration and hashes with published tables. Preserve train/validation/test isolation and fit normalization or projection parameters on training rows only.

Pull requests should describe the resulting behavior, relevant tradeoffs and checks performed. Use merge commits when bringing together independently authored histories so contributor commits remain reachable.
