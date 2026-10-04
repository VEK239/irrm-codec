# DATA-ANCHOR latent benchmark preregistration

Status: locked before inspecting any result from the evaluations defined here.

## Claim under test

At the same 128-dimensional bottleneck and encoder architecture, the frozen
R+T+P representation is a better *balanced, multipurpose representation* than
any frozen single-objective representation. This is not a claim that R+T+P must
win every specialist task or every unrelated downstream endpoint.

The matched seed-42 DATA-ANCHOR models are R, T, P, RT, RP, TP, and RTP. All
seven use the locked TRB benchmark, train-only normalizer, tokenizer, model
architecture, and validation-selected checkpoint. No encoder is tuned using an
evaluation result.

## Primary evaluation: frozen latent sufficiency

Identical fresh readouts are fitted from each frozen latent representation using
the locked training split, selected using validation only, and evaluated once on
test. Readouts measure sequence content, standardized TCRemP target recovery,
and `log10_pgen_1mm` recovery. The original task heads are not reused as probes.

Primary model-level endpoints, in order, are:

1. worst-task normalized regret across R, T, and P targets (lower is better);
2. mean normalized regret across the same targets (lower is better);
3. Pareto hypervolume with a fixed null-reference point (higher is better);
4. number of targets within 5% of the corresponding best specialist.

Task losses are oriented so lower is better and normalized using fixed trivial
train-derived nulls and the best validation-eligible specialist. The test set
does not set signs, weights, thresholds, readout capacity, or normalization.
Paired test-clonotype bootstrap intervals are reported for RTP minus every
comparator. These intervals do not measure training-seed variability.

## Controlled one-amino-acid mutation challenge

Valid single-substitution variants are grouped by parent clonotype. A parent's
variants cannot cross fitting/evaluation folds. TCRemP and pgen effects must be
computed by an existing validated scorer; absence of such a scorer blocks that
endpoint rather than allowing a proxy.

Primary endpoints are balanced prediction of delta-TCRemP and delta-log-pgen,
followed by pgen direction accuracy and mutant-effect rank correlation. Mutation
position/substitution recovery and triplet consistency are secondary. Since all
pairs have edit distance one, this analysis tests organization beyond raw edit
distance. Parent-clustered bootstrap intervals are required.

## Secondary evaluations

These are reported separately and cannot replace a failed primary endpoint:

- nested low-label readout curves;
- fixed subgroup performance by CDR3 length, V/J family, and pgen quantile;
- joint sequence/TCRemP/pgen retrieval using train-defined scaling;
- masking and conservative/nonconservative substitution robustness;
- synthetic-versus-observed discrimination only if a provenance-verified,
  benchmark-deduplicated synthetic source already exists.

## Baselines and efficiency

Every feasible track includes the seven frozen models plus R|T|P concatenation
(384 dimensions), train-only PCA of that concatenation to 128 dimensions, and
sequence composition/length controls (with V/J reported separately). Runtime,
encoder count, representation dimension, and parameter count are reported. The
384-dimensional concatenation is an expensive upper bound, not a size-matched
single-encoder baseline.

## Decision rule

The primary claim is supported only if RTP improves the preregistered balanced
endpoint over each single-objective model with a paired 95% interval excluding
no improvement, while remaining competitive on every constituent task. Wins on
an isolated secondary metric, subgroup, perturbation type, or retrospectively
chosen endpoint do not establish the claim. All negative results are retained.

## Execution controls

All remote computation runs through CPU Slurm using
`/home/evlasova/.conda/envs/irrm-codec/bin/python`. No computation runs on an
Aldan frontend. Inputs, model checkpoints, configs, scripts, and outputs receive
SHA-256 manifests; outputs are isolated by study and Slurm job identifier.
