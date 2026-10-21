# Validation data

Adam, Bill and Eve are synthetic personas. Environmental readings are measured
building inputs; comfort scores and language feedback are synthetic. No real
occupant feedback or user database is included.

## Persona records

- `personas/*_training.csv`: exact analysis pairs, timestamps, saved interpreted
  comfort, sampled numerical comfort, and the matching original feedback text.
- `personas/*_generated_feedback.md`: the original generated logs (60 Adam,
  24 Bill and 8 Eve comments). Fitted models use 60, 23 and 8 respectively;
  `feedback_entry_n` records which log entries were selected.
- `personas/*_model.pt`: final IEW + GP model states used for Table 1. They contain
  tensors, training data and numerical settings, with internal paths removed.
- `personas/*_evaluation.npz`: the fixed 2,500-point temperature/CO2 grid, two
  synthetic references, model means and latent GP variances.
- `personas/*_score_comparison.npz`: the interpreted-score and numerical-score
  comparison on the same evaluation targets.
- `personas/*_learning.npz`, `*_score_bounds.npz`: numerical history used to
  render the learning and comfort-bound figures without retraining.
- `personas/table1_metrics.csv`, `table1.tex`: the paper's five-model comparison.
  `score_source_comparison_iew.tex` is the paired score-source table.

CSV field definitions: `n` is the selected training observation number;
`temperature_F, co2_ppm` are the inputs used for fitting; `days_since_feedback`
is the age at the final fit; `llm_checkpoint_comfort` is the stored training
target; `sampled_numerical_comfort` is the score used by the language generator;
`feedback` is the generated comment. `generation_temperature_F` preserves the
original generation input. `manually_corrected` and
`temperature_aligned_to_table1` identify the changes described below.
`llm_severity_reconstructed = 1 - llm_checkpoint_comfort`; it is not a saved raw
API response. Other comparison columns preserve the existing alignment record.

### Corrections and historical limitations

**Adam observation 40 was manually changed from comfort 0.1 to 1.0.** The paper
model and the bundled retraining inputs retain that correction. Thus these
targets must not be described as entirely unmodified LLM outputs.

Eight Adam temperatures differ between the generation logs and Table 1 inputs
(observations 5, 8, 45, 46, 47, 48, 49 and 60). The numerical-score comparison
uses the Table 1 inputs while retaining the originally sampled scores; scores
were not resampled at the adjusted coordinates. This limits interpretation of
the comparison as a pure language-error experiment.

Three analysis inputs also differ from the currently available hourly source:

| Record | Local timestamp | Current hourly data (F, ppm) | Analysis input (F, ppm) |
|---|---|---|---|
| Adam 53 | 2025-02-10 17:00 | 74.0908, 805.6213 | 66.5462, 431.9360 |
| Bill 8 | 2024-09-15 16:00 | 76.1333, 474.6987 | 79.4478, 481.3707 |
| Eve 8 | 2025-03-05 14:00 | 72.7125, 956.9280 | 72.7125, 1256.9280 |

Their historical cause is not established. Saved analysis records are used for
reproducing the reported results rather than silently replacing these values.

## Environmental series

`environment/selected_hourly_environment.csv` includes the selected channels
only, avoiding unrelated zones and large duplicate source exports:

- `Z25+26+27_temp` from `Zone_temperature_060124-030725_final.csv`, degrees F.
- `Z25+26+27_CO2` from `Zone_CO2_060124-030725_final.csv`, ppm.

Original timestamps were parsed as UTC, converted to US/Eastern and averaged
hourly, June 1, 2024 through March 7, 2025 inclusive. The original median sampling
interval was 60 seconds. The 6,721 hourly rows include the repeated daylight-saving
hour. Synthetic reporting considers 08:00–18:00 inclusive. There are 413 missing
temperature hourly means, of which 190 fall in reporting hours; no CO2 hourly
means are missing. Missing values are retained. The historical filling/dropping
policy and exact public dataset release/DOI were not recovered. The new generator
requires an explicit `skip` policy when missing reporting-hour inputs occur.

## Fixed and variable components

| Stage | Reproduction policy |
|---|---|
| Historical persona generation | Original numerical random state was not recorded; stored logs are reused. |
| Language generation/interpretation | GPT-4o, temperature 0.7; no API seed. Stored training scores are reused for paper comparisons. Full historical API response metadata is not available. |
| Persona fitting | CPU float32, fixed parameter initialization, full-batch Adam, 2,000 steps at learning rate 0.001, tau 120 days, seed 0. Eve starts at 71–74 F; Adam/Bill at 71–77 F. |
| Reference comparison | Stored evaluation grid and residual realization are shared across models within each persona; residual seed 0. |
| Temporal simulations | Ten seeds (0–9), stable/gradual/abrupt scenarios, 720 days. The same simulated sequence is paired across forgetting settings. |
| Severe-error simulations | Ten seeds (0–9), stable latent preference. The first three reported scores in days 300–359 are replaced by the opposite endpoint (0 or 1). |

The original persona generator draws IEW + a GP residual sample + independent
Gaussian observation noise, then clips the sampled score to [0, 1]. The historical
generation loop sampled one condition per call, with a fresh residual each time;
it did not share the later evaluation-grid residual realization. The latter is
a separate fixed synthetic target, not a recovered historical comfort function.

Temporal/adversarial records contain numerical observations directly and do not
call an LLM. Their full settings are in `experiments/settings.py`. Both use 1,000
optimizer steps per fit, learning rate 0.001, and forgetting half-lives 50, 100,
200, 400 days plus no forgetting. The age-variance amplitude is 0.15. Temporal
code explicitly initializes the global noise after constraint registration;
the original persona fitting code preserves its earlier initialization order.
This difference is retained rather than silently changing the paper experiments.

## Metrics

MAE/RMSE use the raw predictive mean. NLPD uses latent GP variance, without test
observation noise; deterministic IEW models have no NLPD. Uncertainty intervals
are Gaussian and unbounded; the displayed bounded comfort output is a separate
clipped point prediction. Evaluation locations on a fixed grid are not
independent experimental runs. The temporal/severe-error data include actual
independent simulation seeds. Numerical reproducibility depends on the pinned
software environment; fresh language generation is not deterministic.
