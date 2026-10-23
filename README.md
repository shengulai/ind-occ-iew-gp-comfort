# Individual comfort modeling

Learn an individual's comfort from environmental conditions and numerical or
natural-language feedback. The model combines an Indoor Environmental Wellness
(IEW) mean function with a residual Gaussian process (GP).

This repository contains a usable Python interface, GPT-4o interpretation,
synthetic example feedback, and the data/functions used for the paper's
simulation validations. The experiments are a methodological proof of concept;
the language scores have not been independently validated with real occupants.

## Publication status

The accompanying manuscript is currently under peer review. The code and
supporting simulation data are publicly available to support reproducibility,
review, and further methodological development. Public availability of this
repository does not imply that the manuscript has been accepted or that the
framework has been validated with real occupants in buildings.

When referencing or reproducing the revised manuscript's results, record the
exact Git commit or tagged release used. This identifies the relevant code and
data as the repository continues to evolve. The final publication citation will
be added when available.

## Install

Use **Python 3.11**. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[validation,llm]'
```

For modeling only, `python -m pip install -e .` is sufficient. Dependency versions
are pinned to the tested environment. On Windows, activate with
`.venv\Scripts\activate`. The commands below run from the repository root.

## Try it without an API key

The example uses already saved comfort scores. It fits a model and writes a
checkpoint plus predictions at the conditions in `examples/conditions.csv`:

```bash
python -m comfort_model \
  --feedback examples/feedback.csv \
  --conditions examples/conditions.csv \
  --output outputs/example
```

Inputs are **temperature in degrees Fahrenheit**, **CO2 in ppm**, and a comfort
score in **[0, 1]**, where 1 means most comfortable. Feedback CSV columns are
`timestamp, temperature_F, co2_ppm, comfort`. Timestamps must use a consistent
timezone. Add a `feedback` column to use natural-language interpretation.

## Use your own natural-language feedback

Set `OPENAI_API_KEY` in your environment, then add `--interpret` to the example
command and choose a new output folder. This explicitly sends each comment to
GPT-4o and uses the returned score for fitting. The SDK reads the key from the
environment; do not put it in source files. See the
[official Python SDK documentation](https://developers.openai.com/api/reference/python).

```bash
python -m comfort_model \
  --feedback examples/feedback.csv \
  --conditions examples/conditions.csv \
  --interpret \
  --output outputs/new_interpretation
```

The optional `--examples path/to/examples.txt` supplies per-person language
examples. Each new interpretation is saved in `interpreted_feedback.csv`,
including the API response and the numerical score, so subsequent fits can reuse
it without API calls. `comfort = 1 - severity`, rounded as in the original
implementation. Invalid or ambiguous responses raise an error. Generation and
interpretation use GPT-4o with temperature 0.7 and no API seed; fresh replies may
differ. No paper correction is applied to users' new feedback.

## Python interface

```python
from comfort_model import ComfortModel, interpret_feedback

model = ComfortModel(tau_days=120)
model.fit(
    [[68, 500], [73, 800], [79, 1200]],
    [0.50, 0.95, 0.30],
    ages_days=[10, 5, 0],
)
prediction = model.predict([[72, 800]])
print(prediction["comfort"])       # clipped point prediction
print(prediction["lower95"], prediction["upper95"])
model.save("outputs/my_model.pt")
model = ComfortModel.load("outputs/my_model.pt")

# Optional API call, followed by a model update at the measured conditions:
feedback = interpret_feedback("The room feels a little too warm.")
model.update(76, 900, feedback.comfort, elapsed_days=1)
```

`update` retains all observations, advances their ages, appends the new pair, and
refits from the configured initial parameters. `tau_days=None` disables the
age-dependent increase in observation variance. The default prior comfort range
is 71–77 F and 0–900 ppm; set `temperature_bounds` and `co2_upper` to change it.
Inputs are normalized using 60–85 F and 0–1800 ppm. Conditions outside this range
are extrapolations.

`mean`, `variance`, `lower95`, and `upper95` describe the **latent Gaussian
prediction**, without additional test observation noise. They are not clipped.
Only `comfort` is clipped to [0, 1]. A nominal 95% interval is not a guarantee of
95% empirical coverage; the validation code evaluates both coverage and width.

## Reproduce the paper checks and figures

These commands use saved inputs and models, require no API key, and do not retrain:

```bash
python -m experiments.reproduce verify --output outputs/metrics
python -m experiments.reproduce figures --output outputs/figures
```

The first regenerates Table 1 (CSV and LaTeX) and the score-source comparison
(CSV), checks the saved model predictions, and regenerates the 30 temporal and
10 severe-error simulation input sequences. It also verifies the saved PMV
experiment and writes its MAE, RMSE and NLPD table under `outputs/metrics/pmv/`.
The second renders persona
comparisons, learning/bound updates, uncertainty diagnostics, temporal-adaptation
results, and the severe-error experiment. Outputs go into a new/empty folder.

To **fit new models** using the supplied experimental settings:

```bash
python -m experiments.refit personas --output outputs/refit_personas
python -m experiments.refit temporal --output outputs/refit_temporal
python -m experiments.refit adversarial --output outputs/refit_adversarial
```

`personas` fits directly calibrated IEW, constant-mean GP, IEW + GP, and the
sampled-numerical-score IEW + GP comparison. Temporal and severe-error experiments
use seeds 0–9. They can take substantial time. Add `--quick` for a short execution
check, or `--seeds 0` to run a single temporal/adversarial seed. Quick results are
not the paper's results. Full settings are in `experiments/settings.py`.

### Temperature-only PMV experiment

This sensitivity test uses a PMV/PPD-derived reference instead of IEW-derived
comfort. It retains Adam, Bill and Eve's analysis temperatures and timestamps,
recalculates numerical comfort scores, and compares the initial and learned
temperature-only IEW + GP. It does not generate or interpret language.

Verify the archived models and results without retraining or additional PMV
dependencies:

```bash
python -m experiments.test_pmv --verify --output outputs/pmv_check
```

To calculate fresh PMV scores and train all three models:

```bash
python -m pip install -e '.[pmv]'
python -m experiments.test_pmv --output outputs/pmv_refit
```

The default run uses 2,000 training steps per persona. Each run writes training
pairs, model checkpoints, evaluation arrays, settings, and a comparison table
in CSV and Markdown. `--steps 20` is available for a short execution check;
it does not reproduce the reported training budget. Use a new/empty output folder.

See [data/pmv/README.md](data/pmv/README.md) for the reference assumptions,
normalization, software citation, and
[archived results](data/pmv/comparison.md). Adam and Eve improved in MAE and RMSE;
Bill's MAE changed little and RMSE increased. NLPD decreased for all three.
These are fixed synthetic scenarios, not independent real-occupant validation.

For a **new synthetic language dataset** (calls GPT-4o):

```bash
python -m experiments.generate_feedback --persona Adam --seed 0 \
  --missing-policy skip --output outputs/new_feedback.csv
```

This preserves the original generator's sampling/reporting rules and prompt;
it cannot reconstruct the historical language dataset. To train from the new
text, pass that CSV to `python -m comfort_model ... --interpret`. To train directly
from its sampled values, rename `sampled_comfort` to `comfort` in a copy.

## Files

| Folder | Contents |
|---|---|
| `comfort_model/` | Model API, original GP equations, interpreter and prompts |
| `examples/` | Small feedback and prediction CSVs |
| `experiments/` | Settings, validation, fitting, simulation and figure functions |
| `data/` | Selected environmental series, synthetic feedback, training pairs, saved models and numerical results |
| `tests/` | Offline interface, checkpoint and interpretation checks |

See [data/README.md](data/README.md) for data provenance, the Adam correction,
score definitions, and what is reproducible. The repository has no embedded API
keys, machine-specific paths, internal tracking JSON, or user database. Outputs,
local environments and credentials are excluded by `.gitignore`.

```bash
python -m unittest discover -s tests -v
```

Packaging checks exercised the offline example, model save/load/update, mocked
API interpretation, saved-result verification, figure generation, and quick
refits. Fresh paid API calls and all full-length training runs were not repeated
as part of the initial packaging. The PMV addition was checked through offline
verification and a fresh full-length fit of all three personas. On macOS systems
with an OpenMP shared-memory error, prefix
commands with `KMP_USE_SHM=0 OMP_NUM_THREADS=1`.

## Release information

The code/data license remains to be specified by the authors. The final
publication citation is pending publication; until then, describe the
accompanying manuscript as under review.
