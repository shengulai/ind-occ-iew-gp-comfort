"""Small CPU interface for fitting, updating, and saving individual comfort models."""
from contextlib import redirect_stdout, nullcontext
from copy import deepcopy
import io
import math
from pathlib import Path

import numpy as np
import torch
import gpytorch

from .gp import ComfortBaseline, ResidualGP, fit_model


class ComfortModel:
    """Inputs are [temperature in degrees F, CO2 in ppm]; comfort is in [0, 1].

    ``tau_days`` is the half-life of the age-dependent noise increase. ``None``
    disables that increase. All observations are retained when updating.
    """

    def __init__(self, temperature_bounds=(71., 77.), co2_upper=900.,
                 tau_days=120., steps=2000, learning_rate=.001):
        lo, hi = map(float, temperature_bounds)
        if not 60 <= lo < hi <= 85 or not 0 < co2_upper <= 1800:
            raise ValueError("Initial bounds must lie within 60–85 F and 0–1800 ppm.")
        if tau_days is not None and (not np.isfinite(tau_days) or tau_days <= 0):
            raise ValueError("tau_days must be positive, or None for no forgetting.")
        if not isinstance(steps, int) or steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Require a positive integer steps and positive learning_rate.")
        self.options = dict(temperature_bounds=(lo, hi), co2_upper=float(co2_upper),
                            tau_days=tau_days, steps=steps, learning_rate=learning_rate)
        self.config = dict(weights_initial={'temp': 1., 'co2': 1.},
                           wellness_range_initial={'temp': [lo, hi, 60., 85.],
                                                   'co2': [0., float(co2_upper), 0., 1800.]},
                           learn_weights_ini=False, fixed_lower_bounds_ini={'co2': 0.},
                           normalize_inputs=True, occ_rate=1., noise_llm_low=.001,
                           noise_llm_high=.05, signal_var_low=.001, signal_var_high=.1,
                           lengthscale_low=.02, lengthscale_high=.2, lengthscale_init=.1)
        self.model = None

    @staticmethod
    def _inputs(conditions):
        x = np.asarray(conditions, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("conditions must be a nonempty, finite N x 2 array [temperature_F, co2_ppm].")
        if np.any(x[:, 1] < 0):
            raise ValueError("CO2 cannot be negative.")
        return x

    def fit(self, conditions, comfort, *, ages_days=None, verbose=False):
        """Refit from the configured initial parameters using all supplied pairs.

        Ages are measured at the current update time. Omitted ages mean all
        observations are treated as current. No LLM is called by this method.
        """
        x = self._inputs(conditions)
        y = np.asarray(comfort, dtype=np.float32)
        ages = np.zeros(len(x)) if ages_days is None else np.asarray(ages_days, dtype=float)
        if y.shape != (len(x),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError("comfort must contain one finite value in [0, 1] per condition.")
        if ages.shape != y.shape or not np.isfinite(ages).all() or np.any(ages < 0):
            raise ValueError("ages_days must contain one finite, nonnegative age per observation.")
        c = self.config
        with nullcontext() if verbose else redirect_stdout(io.StringIO()):
            model, likelihood, _ = fit_model(
                torch.tensor(x), torch.tensor(y), weights_ini=deepcopy(c['weights_initial']),
                wellness_range_ini=deepcopy(c['wellness_range_initial']),
                learn_weights_ini=c['learn_weights_ini'], fixed_lower_bounds_ini=c['fixed_lower_bounds_ini'],
                normalize_inputs=c['normalize_inputs'], occ_rate=c['occ_rate'],
                days_since_feedback=ages, tau=self.options['tau_days'] or math.inf,
                num_steps=self.options['steps'], lr=self.options['learning_rate'],
                **{k: c[k] for k in ('noise_llm_low', 'noise_llm_high', 'signal_var_low',
                                     'signal_var_high', 'lengthscale_low', 'lengthscale_high', 'lengthscale_init')})
        model.eval()
        likelihood.eval()
        self.model, self.ages_days = model, ages.copy()
        return self

    def predict(self, conditions):
        """Return raw Gaussian mean/variance, latent 95% interval, and clipped comfort.

        ``comfort`` is the clipped point output. The Gaussian interval and raw
        mean remain unbounded and exclude test observation noise.
        """
        if self.model is None:
            raise RuntimeError("Fit or load a model before prediction.")
        x = torch.tensor(self._inputs(conditions))
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            latent = self.model(x)
            mean, variance = latent.mean.numpy(), latent.variance.numpy()
        width = 1.959963984540054 * np.sqrt(variance)
        return dict(comfort=np.clip(mean, 0., 1.), mean=mean, variance=variance,
                    lower95=mean-width, upper95=mean+width)

    def update(self, temperature_F, co2_ppm, comfort, *, elapsed_days=0., verbose=False):
        """Append a new observation, age previous observations, and refit."""
        if not np.isfinite(elapsed_days) or elapsed_days < 0:
            raise ValueError("elapsed_days must be finite and nonnegative.")
        x = [[temperature_F, co2_ppm]]
        y, ages = [comfort], [0.]
        if self.model is not None:
            x = np.vstack([self.model.train_inputs[0].numpy(), x])
            y = np.r_[self.model.train_targets.numpy(), y]
            ages = np.r_[self.ages_days + elapsed_days, 0.]
        return self.fit(x, y, ages_days=ages, verbose=verbose)

    def save(self, path):
        """Save tensors and numerical configuration; no API credentials or prompts."""
        if self.model is None:
            raise RuntimeError("Fit a model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state_dict=self.model.state_dict(),
                        X=self.model.train_inputs[0], y=self.model.train_targets,
                        fixed_noise=self.model.likelihood.noise_covar.noise,
                        config=self.config, options=self.options,
                        ages_days=torch.tensor(self.ages_days)), path)

    @classmethod
    def load(cls, path):
        """Load a state-only checkpoint with PyTorch's restricted weights loader."""
        saved = torch.load(path, map_location='cpu', weights_only=True)
        obj = cls(**saved['options'])
        c = obj.config = saved['config']
        baseline = ComfortBaseline(c['weights_initial'], c['wellness_range_initial'],
                                   learn_weights=c['learn_weights_ini'],
                                   fixed_lower_bounds=c['fixed_lower_bounds_ini'],
                                   normalize_inputs=c['normalize_inputs'])
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=saved['fixed_noise'], learn_additional_noise=True)
        likelihood.second_noise_covar.register_constraint(
            'raw_noise', gpytorch.constraints.Interval(c['noise_llm_low'], c['noise_llm_high']))
        model = ResidualGP(saved['X'], saved['y'], likelihood, baseline, occ_rate=c['occ_rate'],
                           **{k: c[k] for k in ('signal_var_low', 'signal_var_high', 'lengthscale_low',
                                                'lengthscale_high', 'lengthscale_init')})
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model.eval()
        likelihood.eval()
        obj.model, obj.ages_days = model, saved['ages_days'].numpy()
        return obj
