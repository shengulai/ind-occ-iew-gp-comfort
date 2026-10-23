# Temperature-only PMV/PPD reference experiment

This experiment tests IEW + GP learning against a synthetic reference with a
different functional form. It compares an unconditioned initial GP with a final
learned GP. The temperature and timestamp sequence is held fixed; no language
generation, interpretation, sampled GP residual or observation noise is added
to the numerical targets.

## Reference

Scores use the unrounded ISO PMV/PPD implementation in **pythermalcomfort 4.6.2**
(`pmv_ppd_iso`, model `7730-2025`):

```text
PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
C   = (100 - PPD) / 95
```

PPD has a minimum of 5% at PMV = 0, so this normalization makes neutral PMV
correspond to C = 1. It is a synthetic comfort benchmark, not a validated
individual comfort probability.

Air temperature approximates mean radiant temperature. Relative humidity is
50%, relative air speed 0.1 m/s, metabolic rate 1.1 met, and external work 0 met.
A fixed clothing value is solved for each persona to make PMV zero at the
midpoint of their predefined comfort bounds. Their original interval widths
are not imposed on the PMV curve.

| Persona | Original bounds (F) | Neutral temperature (F) | Clothing (clo) |
|---|---|---:|---:|
| Adam | 70–74 | 72.0 | 1.032079 |
| Bill | 74–77 | 75.5 | 0.703281 |
| Eve | 72–76 | 74.0 | 0.843619 |

All final training and evaluation conditions pass the implementation's PMV
applicability checks. Invalid conditions raise an error instead of being dropped.

## Inputs and fitting

- The original analysis temperatures and timestamps are read from
  `data/personas/*_training.csv`: 60 Adam, 23 Bill and 8 Eve observations.
  Historical input selection and alignment are documented in [../README.md](../README.md).
- Every target is recalculated from PMV, including Adam observation 40.
  No historical LLM score or manual score correction is used as a PMV target.
- The IEW mean and GP have only a temperature component, normalized using
  60–85 F. Initial IEW comfort bounds are 71–77 F for Adam/Bill and 71–74 F
  for Eve. These are the original fitted-model initializations.
- Each model is fitted once to all its observations from that initialization:
  CPU float32, one thread, seed 0, full-batch Adam, 2,000 steps, learning rate
  0.001. Ages are calendar days relative to the last observation.
- The original Gaussian observation model is retained even though these targets
  have no added noise. Fixed observation variance is
  `0.005 + 0.15 * (1 - exp(-log(2) * age_days / 120))`, plus a learned global
  variance constrained to 0.001–0.05. The original likelihood initialization
  order is preserved.
- Kernel signal variance starts at 0.005, with bounds 0.001–0.1 and the original
  Gamma prior; normalized length scale starts at 0.1, with bounds 0.02–0.2.
  Full settings are in `experiments/settings.py`.

## Evaluation and interpretation

Both models use the same 501-point uniform grid from 66 to 80 F. MAE and RMSE
use the raw predictive means. NLPD uses marginal Gaussian **latent** predictive
variance, excluding test observation noise. The initial GP is unconditioned,
with variance 0.005; the learned GP uses its posterior variance. No clipping
or variance recalibration is applied.

The [saved comparison](comparison.md) shows reduced MAE/RMSE for Adam and Eve;
Bill's MAE changes little and RMSE increases. NLPD decreases for all three,
which does not by itself establish calibrated uncertainty. These are three
fixed synthetic scenarios, not independent simulation replications or a test
of PMV-based feedback-reporting behavior. Their one-dimensional reference and
evaluation domain differ from the two-dimensional Table 1 experiment.

## Included files and commands

- `*_training_pairs.csv`: selected temperatures, timestamps, ages, PMV, PPD,
  normalized targets and their float32 training values.
- `*_pmv_gp.ckpt`: fitted model and likelihood states, training tensors and
  numerical configuration. `experiments.test_pmv.load_checkpoint` loads these
  temperature-only models; the main two-factor `ComfortModel` API is unchanged.
- `*_evaluation.npz`: fixed grid, PMV/PPD reference, initial/learned means and
  latent variances, training inputs, targets, timestamps and fixed noise.
- `reference_settings.csv`: reference and fitted parameters.
- `metrics.csv`, `comparison.md`: numerical and readable results.

From the repository root, `python -m experiments.test_pmv --verify --output
outputs/pmv_check` checks saved predictions and recomputes metrics without
retraining or importing pythermalcomfort. For fresh targets and training,
install `python -m pip install -e '.[pmv]'`, then run
`python -m experiments.test_pmv --output outputs/pmv_refit`.
Fresh runs write separate files and leave the archived results intact.

## Software citation

Tartarini, F., and Schiavon, S. (2020). *pythermalcomfort: A Python package for
thermal comfort research*. SoftwareX, 12, 100578.
[https://doi.org/10.1016/j.softx.2020.100578](https://doi.org/10.1016/j.softx.2020.100578).

The paper describes the software; the pinned version above identifies the
implementation used for this experiment.
