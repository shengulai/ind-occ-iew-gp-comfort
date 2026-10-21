from copy import deepcopy
import math
import csv
import numpy as np
import torch
import gpytorch
from comfort_model.gp import ComfortBaseline, ResidualGP, fit_model

def new_baseline(cfg):
    return ComfortBaseline(weights=deepcopy(cfg['weights_initial']),
                           wellness_range=deepcopy(cfg['wellness_range_initial']),
                           learn_weights=cfg['learn_weights_ini'],
                           fixed_lower_bounds=cfg['fixed_lower_bounds_ini'],
                           normalize_inputs=cfg['normalize_inputs'])


def paper_default_baseline(cfg):
    """Fixed manuscript comparator, distinct from fitted-model initialization."""
    initial_cfg = deepcopy(cfg)
    initial_cfg['wellness_range_initial'] = {'temp':[72,76,60,85], 'co2':[0,900,0,1800]}
    return new_baseline(initial_cfg)


def fit_iew(X, y, cfg, steps, learning_rate):
    """Fit only IEW parameters. No ages, weights, noise model, or GP residual."""
    baseline = new_baseline(cfg)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=learning_rate)
    initial = float(torch.mean((baseline(X, cfg['occ_rate'])-y)**2).detach())
    for step in range(steps):
        optimizer.zero_grad()
        loss = torch.mean((baseline(X, cfg['occ_rate'])-y)**2)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite IEW loss at step {step}; no observations were discarded.')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in baseline.parameters()):
            raise RuntimeError(f'Nonfinite IEW gradient at step {step}.')
        optimizer.step()
    optimizer.zero_grad()
    final = torch.mean((baseline(X, cfg['occ_rate'])-y)**2)
    if not torch.isfinite(final):
        raise RuntimeError('Nonfinite final IEW loss.')
    final.backward()
    grad_norm = math.sqrt(sum(float((p.grad**2).sum()) for p in baseline.parameters() if p.grad is not None))
    baseline.eval()
    return baseline, dict(iew_initial_mse=initial, iew_final_mse=float(final.detach()), iew_gradient_norm=grad_norm)


def fit_proposed(X, y, ages, cfg, a):
    # Preserve production initialization and likelihood behavior; do not silently
    # substitute the modified likelihood initialization in the forgetting test.
    return fit_model(
        X, y, weights_ini=deepcopy(cfg['weights_initial']),
        wellness_range_ini=deepcopy(cfg['wellness_range_initial']),
        learn_weights_ini=cfg['learn_weights_ini'], fixed_lower_bounds_ini=cfg['fixed_lower_bounds_ini'],
        normalize_inputs=cfg['normalize_inputs'], occ_rate=cfg['occ_rate'],
        days_since_feedback=ages, tau=a.tau, num_steps=a.steps, lr=a.learning_rate,
        noise_llm_init=a.base_variance, noise_hist_high=a.history_variance,
        noise_llm_low=cfg['noise_llm_low'], noise_llm_high=cfg['noise_llm_high'],
        signal_init_var=.005, signal_var_low=cfg['signal_var_low'], signal_var_high=cfg['signal_var_high'],
        lengthscale_low=cfg['lengthscale_low'], lengthscale_high=cfg['lengthscale_high'],
        lengthscale_init=cfg['lengthscale_init'])


def fit_constant_gp(X, y, ages, cfg, a):
    """Same kernel/noise as production, learned constant mean instead of IEW."""
    normalizer = new_baseline(cfg)
    normalizer.requires_grad_(False)
    fixed = a.base_variance + a.history_variance*(1-torch.exp(-math.log(2)*torch.tensor(ages,dtype=X.dtype)/a.tau))
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed, learn_additional_noise=True)
    # Match production's assignment BEFORE constraint registration, including
    # its effective transformed initial value, rather than changing one model.
    with torch.no_grad():
        likelihood.second_noise = torch.tensor(cfg['noise_llm_low'],dtype=X.dtype)
    likelihood.second_noise_covar.register_constraint('raw_noise',gpytorch.constraints.Interval(cfg['noise_llm_low'],cfg['noise_llm_high']))
    model = ResidualGP(X,y,likelihood,normalizer,occ_rate=cfg['occ_rate'],
                      signal_var_low=cfg['signal_var_low'],signal_var_high=cfg['signal_var_high'],
                      lengthscale_low=cfg['lengthscale_low'],lengthscale_high=cfg['lengthscale_high'],
                      lengthscale_init=cfg['lengthscale_init'])
    # The frozen baseline now provides normalization ONLY; its forward output
    # and comfort bounds play no role in the GP mean or parameter fitting.
    model.mean_module = gpytorch.means.ConstantMean(constant_constraint=gpytorch.constraints.Interval(0.,1.))
    model.mean_module.initialize(constant=.5)
    model.covar_module.outputscale = .005
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params':model.mean_module.parameters()},
        {'params':model.covar_module.parameters()},
        {'params':likelihood.parameters()}],lr=a.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
    for step in range(a.steps):
        optimizer.zero_grad()
        loss = -mll(model(X),y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite constant-mean GP loss at step {step}.')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Nonfinite constant-mean GP gradient at step {step}.')
        optimizer.step()
    return model, likelihood


def score(prediction, reference):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid prediction/reference shape or nonfinite values.')
    error = prediction-reference
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))))


def bounds_of(model):
    return np.array([float(v.detach()) for factor in ['temp', 'co2'] for v in model.get_comfort_bounds(factor)])


def rbf_root(coordinates, lengthscale):
    """Symmetric square root of a unit-variance RBF covariance, in float64."""
    distance = (coordinates[:, None]-coordinates[None, :])/lengthscale
    covariance = np.exp(-0.5*distance**2)
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-10:
        raise ValueError('RBF covariance has a materially negative eigenvalue.')
    return (vectors*np.sqrt(np.maximum(values, 0))) @ vectors.T


def draw_evaluation_residual(grid, params, seed):
    """One correlated GP draw on a Cartesian grid, never conditioned on training.

    The ARD RBF kernel is separable: K = signal_var * K_temp (x) K_co2.
    Factoring the two small covariance matrices avoids an approximate large-GP
    sampler. Roundoff-negative eigenvalues are clipped to zero; no noise is added.
    """
    grid = np.asarray(grid, dtype=np.float64)
    temp, ti = np.unique(grid[:, 0], return_inverse=True)
    co2, ci = np.unique(grid[:, 1], return_inverse=True)
    if len(grid) != len(temp)*len(co2) or len(np.unique(grid, axis=0)) != len(grid):
        raise ValueError('Residual sampling requires a complete Cartesian evaluation grid.')
    keys = ['signal_var', 'lengthscale_temp', 'lengthscale_co2']
    if any(not math.isfinite(params[k]) or params[k] <= 0 for k in keys):
        raise ValueError('Residual signal variance and length scales must be finite and positive.')
    rt = rbf_root(temp, params['lengthscale_temp'])
    rc = rbf_root(co2, params['lengthscale_co2'])
    z = np.random.default_rng(seed).standard_normal((len(temp), len(co2)))
    field = math.sqrt(params['signal_var']) * rt @ z @ rc.T
    return field[ti, ci]
