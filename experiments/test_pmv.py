"""Temperature-only PMV/PPD sensitivity test; no language API calls.

Verify saved results: python -m experiments.test_pmv --verify --output outputs/pmv_check
Fit new models:      python -m experiments.test_pmv --output outputs/pmv_refit
Only fresh fitting requires the optional PMV dependencies (pip install -e '.[pmv]').
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "experiments"

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import random
from types import SimpleNamespace

import gpytorch
import numpy as np
import torch

from comfort_model.gp import ResidualGP
from .baselines import fit_proposed, new_baseline
from .metrics import scores
from .settings import PERSONAS, INITIAL, REFERENCES, PMV

DATA = Path(__file__).resolve().parents[1] / "data"
NLPD_NOTE = (
    "NLPD uses marginal Gaussian latent predictions, without test observation noise. "
    "Initial variance is 0.005; learned variance is the posterior latent variance. "
    "Means and variances are not clipped or recalibrated. Lower NLPD is better, "
    "including negative values; NLPD alone does not establish interval calibration."
)


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def reference_score(temperature_f, clo, *, check_limits=True):
    """Unrounded ISO PMV/PPD; no optional library import during saved verification."""
    from pythermalcomfort.models import pmv_ppd_iso
    temperature_c = (np.asarray(temperature_f, dtype=float) - 32) * 5 / 9
    result = pmv_ppd_iso(tdb=temperature_c, tr=temperature_c, clo=clo,
                         limit_inputs=check_limits, **PMV["reference_options"])
    pmv, ppd = np.asarray(result.pmv), np.asarray(result.ppd)
    if not np.isfinite(pmv).all() or not np.isfinite(ppd).all():
        raise ValueError("PMV applicability check failed; no observations are silently discarded.")
    comfort = (100 - ppd) / 95
    np.testing.assert_allclose(comfort, np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2),
                               rtol=1e-12, atol=1e-14)
    if np.any((comfort < 0) | (comfort > 1)):
        raise ValueError("PMV-derived comfort outside [0, 1].")
    return pmv, ppd, comfort


def clothing_at_neutrality(neutral_f):
    from scipy.optimize import brentq
    # Bracket probes may exceed |PMV|=2; all final reference scores are checked.
    objective = lambda clo: float(reference_score(neutral_f, clo, check_limits=False)[0])
    clo = float(brentq(objective, *PMV["clothing_bounds"], xtol=1e-10))
    pmv, _, comfort = reference_score(neutral_f, clo)
    np.testing.assert_allclose(pmv, 0, atol=1e-7)
    np.testing.assert_allclose(comfort, 1, atol=1e-12)
    return clo


def temperature_config(persona):
    cfg = deepcopy(INITIAL[persona])
    cfg["weights_initial"] = {"temp": cfg["weights_initial"]["temp"]}
    cfg["wellness_range_initial"] = {"temp": cfg["wellness_range_initial"]["temp"]}
    cfg["fixed_lower_bounds_ini"] = {}
    return cfg


def load_inputs(persona):
    pairs = read_rows(DATA / f"personas/{persona}_training.csv")
    inputs = np.array([[float(r["temperature_F"])] for r in pairs], dtype=np.float32)
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in pairs]
    if stamps != sorted(stamps) or [int(r["n"]) for r in pairs] != list(range(1, len(pairs) + 1)):
        raise ValueError(f"{persona}: unexpected observation order.")
    ages = np.array([(stamps[-1].date() - stamp.date()).days for stamp in stamps])
    np.testing.assert_array_equal(ages, [int(r["days_since_feedback"]) for r in pairs])
    if not np.isfinite(inputs).all():
        raise ValueError(f"{persona}: nonfinite training inputs.")
    return pairs, inputs, ages


def initial_prediction(cfg, grid):
    baseline = new_baseline(cfg).eval()
    prior = ResidualGP(None, None, gpytorch.likelihoods.GaussianLikelihood(), baseline,
                       occ_rate=cfg["occ_rate"], **{key: cfg[key] for key in
                       ("signal_var_low", "signal_var_high", "lengthscale_low", "lengthscale_high", "lengthscale_init")})
    prior.covar_module.outputscale = .005
    prior.eval()
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        prediction = prior(grid)
        mean, variance = prediction.mean.numpy().copy(), prediction.variance.numpy().copy()
        np.testing.assert_array_equal(mean, baseline(grid, cfg["occ_rate"]).numpy())
    np.testing.assert_allclose(variance, .005, rtol=1e-6)
    return mean, variance


def save_checkpoint(path, model, config):
    cfg = deepcopy(config)
    # likelihood.noise includes the learned additional variance; saving that as
    # fixed noise would add it twice on reload. Export only noise_covar.noise.
    cfg.update(likelihood_type="fixed_noise",
               fixed_noise_vec=model.likelihood.noise_covar.noise.detach().cpu().clone())
    torch.save(dict(model_state_dict=model.state_dict(),
                    likelihood_state_dict=model.likelihood.state_dict(),
                    X_train_final=model.train_inputs[0].detach().cpu(),
                    y_train_final=model.train_targets.detach().cpu(), config=cfg), path)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    if list(cfg["weights_initial"]) != ["temp"]:
        raise ValueError("Expected a temperature-only PMV experiment checkpoint.")
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=cfg["fixed_noise_vec"], learn_additional_noise=True)
    likelihood.second_noise_covar.register_constraint(
        "raw_noise", gpytorch.constraints.Interval(cfg["noise_llm_low"], cfg["noise_llm_high"]))
    model = ResidualGP(saved["X_train_final"], saved["y_train_final"], likelihood,
                       new_baseline(cfg), occ_rate=cfg["occ_rate"], **{key: cfg[key] for key in
                       ("signal_var_low", "signal_var_high", "lengthscale_low", "lengthscale_high", "lengthscale_init")})
    model.load_state_dict(saved["model_state_dict"], strict=True)
    likelihood.load_state_dict(saved["likelihood_state_dict"], strict=True)
    model.eval()
    likelihood.eval()
    return model, saved


def metric_row(persona, count, arrays):
    result = dict(persona=persona, n_observations=count)
    for arm in ("initial", "learned"):
        mean, variance = arrays[f"{arm}_mean"], arrays[f"{arm}_latent_variance"]
        if any(v.shape != arrays["reference"].shape or not np.isfinite(v).all()
               for v in (mean, variance)) or np.any(variance <= 0):
            raise ValueError("Invalid predictive mean or latent variance.")
        result.update({f"{arm}_{key}": value for key, value in scores(mean, variance, arrays["reference"]).items()})
    return result


def write_comparison(out, rows):
    keys = ("initial_mae", "learned_mae", "initial_rmse", "learned_rmse", "initial_nlpd", "learned_nlpd")
    lines = ["| Persona | Observations | Initial MAE | Learned MAE | Initial RMSE | Learned RMSE | Initial NLPD | Learned NLPD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['persona']} | {row['n_observations']} | " +
                     " | ".join(f"{row[key]:.4f}" for key in keys) + " |")
    write_rows(out / "metrics.csv", rows)
    (out / "comparison.md").write_text("\n".join(lines) + "\n\n" + NLPD_NOTE + "\n")
    print("\n".join(lines), flush=True)


def verify_saved(out, archive=None):
    """Recompute metrics/check predictions, without fitting or PMV dependencies."""
    archive = DATA / "pmv" if archive is None else Path(archive)
    expected = {r["persona"]: r for r in read_rows(archive / "metrics.csv")}
    results = []
    for persona in PERSONAS:
        pairs, inputs, ages = load_inputs(persona)
        arrays = read_arrays(archive / f"{persona}_evaluation.npz")
        np.testing.assert_array_equal(arrays["training_temperature_F"], inputs[:, 0])
        np.testing.assert_array_equal(arrays["timestamps"], [p["timestamp"] for p in pairs])
        np.testing.assert_array_equal(arrays["age_days"], ages)
        np.testing.assert_allclose(arrays["reference"], (100 - arrays["ppd"]) / 95, rtol=0, atol=1e-14)
        np.testing.assert_allclose(arrays["reference"],
                                   np.exp(-.03353 * arrays["pmv"]**4 - .2179 * arrays["pmv"]**2), rtol=1e-12)
        grid = torch.from_numpy(arrays["temperature_F"].copy()).reshape(-1, 1)
        mean, variance = initial_prediction(temperature_config(persona), grid)
        np.testing.assert_array_equal(mean, arrays["initial_mean"])
        np.testing.assert_array_equal(variance, arrays["initial_latent_variance"])
        model, saved = load_checkpoint(archive / f"{persona}_pmv_gp.ckpt")
        np.testing.assert_array_equal(saved["X_train_final"].numpy(), inputs)
        np.testing.assert_array_equal(saved["y_train_final"].numpy(), arrays["training_comfort"])
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            prediction = model(grid)
            np.testing.assert_allclose(prediction.mean.numpy(), arrays["learned_mean"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(prediction.variance.numpy(), arrays["learned_latent_variance"], rtol=0, atol=1e-7)
        result = metric_row(persona, len(pairs), arrays)
        for key, value in result.items():
            if key != "persona":
                np.testing.assert_allclose(value, float(expected[persona][key]), rtol=0, atol=1e-12)
        results.append(result)
        print(f"{persona}: PMV inputs, saved predictions and MAE/RMSE/NLPD verified.", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_comparison(out, results)
    return results


def fit_personas(out, steps):
    try:
        actual_version = version("pythermalcomfort")
    except ModuleNotFoundError as error:
        raise RuntimeError("Install PMV dependencies: python -m pip install -e '.[pmv]'") from error
    if actual_version != PMV["package_version"]:
        raise RuntimeError(f"Use pythermalcomfort=={PMV['package_version']} for the pinned reference.")
    settings = dict(PMV["training"], steps=steps)
    grid = torch.linspace(*PMV["temperature_range_F"], PMV["grid_size"]).reshape(-1, 1)
    results, scenarios = [], []
    for persona in PERSONAS:
        torch.manual_seed(settings["seed"])
        np.random.seed(settings["seed"])
        random.seed(settings["seed"])
        pairs, inputs, ages = load_inputs(persona)
        cfg = temperature_config(persona)
        preferred = REFERENCES[persona]["wellness_range"]["temp"][:2]
        neutral_f = sum(preferred) / 2
        clo = clothing_at_neutrality(neutral_f)
        pmv, ppd, targets = reference_score(inputs[:, 0], clo)
        eval_pmv, eval_ppd, reference = reference_score(grid[:, 0].numpy(), clo)
        initial_mean, initial_variance = initial_prediction(cfg, grid)
        X, y = torch.from_numpy(inputs.copy()), torch.tensor(targets, dtype=torch.float32)
        print(f"{persona}: fitting {steps} steps with {len(y)} numerical scores; clo={clo:.6f}", flush=True)
        with (out / f"{persona}_training.log").open("w") as log, redirect_stdout(log):
            model, likelihood, baseline = fit_proposed(X, y, ages.tolist(), cfg, SimpleNamespace(**settings))
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"{persona}: nonfinite fitted parameter.")
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            learned = model(grid)
            mean, variance = learned.mean.numpy().copy(), learned.variance.numpy().copy()
        checkpoint = out / f"{persona}_pmv_gp.ckpt"
        export_cfg = dict(cfg, pmv_clothing_clo=clo, pmv_options=PMV["reference_options"],
                          reference_neutral_F=neutral_f, experiment_settings=settings)
        save_checkpoint(checkpoint, model, export_cfg)
        restored, _ = load_checkpoint(checkpoint)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            check = restored(grid)
            np.testing.assert_allclose(check.mean.numpy(), mean, rtol=0, atol=1e-6)
            np.testing.assert_allclose(check.variance.numpy(), variance, rtol=0, atol=1e-7)
        arrays = dict(temperature_F=grid[:, 0].numpy(), pmv=eval_pmv, ppd=eval_ppd,
                      reference=reference, initial_mean=initial_mean, initial_latent_variance=initial_variance,
                      learned_mean=mean, learned_latent_variance=variance, training_temperature_F=inputs[:, 0],
                      training_comfort=y.numpy(), timestamps=np.array([p["timestamp"] for p in pairs]),
                      age_days=ages, fixed_noise=likelihood.noise_covar.noise.detach().numpy())
        np.savez_compressed(out / f"{persona}_evaluation.npz", **arrays)
        write_rows(out / f"{persona}_training_pairs.csv", [
            dict(persona=persona, n=i+1, feedback_entry_n=int(p["feedback_entry_n"]), timestamp=p["timestamp"],
                 temperature_F=float(inputs[i, 0]), age_days=int(ages[i]), pmv=float(pmv[i]), ppd=float(ppd[i]),
                 pmv_reference_comfort=float(targets[i]), training_comfort_float32=float(y[i]))
            for i, p in enumerate(pairs)])
        results.append(metric_row(persona, len(y), arrays))
        lo, hi = [float(v.detach().item()) for v in baseline.get_comfort_bounds("temp")]
        scenarios.append(dict(persona=persona, neutral_F=neutral_f, clothing_clo=clo,
                              initial_iew_lower_F=cfg["wellness_range_initial"]["temp"][0],
                              initial_iew_upper_F=cfg["wellness_range_initial"]["temp"][1],
                              learned_iew_lower_F=lo, learned_iew_upper_F=hi))
    write_rows(out / "reference_settings.csv", scenarios)
    write_comparison(out, results)
    notes = ["# Fresh PMV experiment", "", NLPD_NOTE, "",
             "Protocol and assumptions: data/pmv/README.md. Original archived results were not modified.",
             "", "## Training settings", ""]
    notes += [f"- {key}: {value}" for key, value in settings.items()]
    notes += ["", "## Software versions", ""]
    notes += [f"- {name}: {version(name)}" for name in
              ("pythermalcomfort", "numba", "llvmlite", "numpy", "scipy", "torch", "gpytorch")]
    (out / "README.md").write_text("\n".join(notes) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--verify", action="store_true", help="Verify saved results; no retraining or PMV dependency.")
    parser.add_argument("--steps", type=int, default=PMV["training"]["steps"], help="Fresh fit budget; default 2000.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new/empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.verify:
        verify_saved(args.output)
    else:
        fit_personas(args.output, args.steps)


if __name__ == "__main__":
    main()
