import csv
import math
import numpy as np
import torch
import gpytorch
from comfort_model.gp import ComfortBaseline, ResidualGP
PRIOR = {"temp": [72.0, 76.0, 60.0, 85.0], "co2": [0.0, 900.0, 0.0, 1800.0]}
Z95 = 1.959963984540054
ANNUAL_PERIOD_DAYS = 365
SEASON_STARTS = (0, 92, 183, 274)
SEASON_NAMES = ("Spring", "Summer", "Fall", "Winter")
SEASON_OFFSETS = np.array([0., 1., 0., -1.])




def noise_variance(ages, tau, base, amplitude):
    """Fixed observation variance. The learned global variance is added separately."""
    weights = np.ones_like(np.asarray(ages, dtype=float)) if math.isinf(tau) else np.exp2(-np.asarray(ages, dtype=float) / tau)
    return base + amplitude * (1.0 - weights)


def variants(a):
    result = [("no_forgetting", math.inf, 0.15)]
    for amplitude in a.history_variances:
        for tau in a.taus:
            result.append((f"tau_{tau:g}_a_{amplitude:g}", tau, amplitude))
    return result


def change_fraction(day, scenario, a):
    if scenario == "stable":
        return 0.0
    if scenario == "abrupt":
        return float(day >= a.change_day)
    if scenario == "gradual":
        return float(np.clip((day - a.change_day) / a.drift_duration, 0, 1))
    raise ValueError(f"No one-way change fraction for scenario {scenario!r}.")


def annual_shape(a):
    # Configurations saved before this option was introduced used a sinusoid.
    return getattr(a, "annual_shape", "sinusoidal")


def annual_offset(day, params, a):
    """Vectorized annual temperature offset; step mode starts in spring."""
    day = np.asarray(day, dtype=float)
    if annual_shape(a) == "seasonal_steps":
        season = np.searchsorted(SEASON_STARTS[1:], np.mod(day, ANNUAL_PERIOD_DAYS), side="right")
        return a.annual_amplitude * SEASON_OFFSETS[season]
    return a.annual_amplitude * np.sin(2*np.pi*day/ANNUAL_PERIOD_DAYS + params["annual_phase"])


def preference_bounds(day, scenario, params, a):
    """Annual preference changes start at day zero; CO2 and width stay fixed."""
    if scenario == "annual":
        center = params["center"] + annual_offset(day, params, a)
        co2_limit = params["co2_limit"]
    else:
        fraction = change_fraction(day, scenario, a)
        center = params["center"] + a.temp_shift * fraction
        co2_limit = params["co2_limit"] + a.co2_shift * fraction
    return center-params["width"]/2, center+params["width"]/2, co2_limit


def reference(X, day, scenario, params, a):
    lower, upper, co2_limit = preference_bounds(day, scenario, params, a)
    if not (60 < lower < upper < 85 and 0 < co2_limit < 1800):
        raise ValueError("Preference shifts moved comfort bounds outside the model's danger bounds.")
    model = ComfortBaseline(
        {"temp": 1.0, "co2": 1.0},
        {"temp": [lower, upper, 60., 85.], "co2": [0., co2_limit, 0., 1800.]},
        learn_weights=False,
    )
    with torch.no_grad():
        base = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
    xn = (X - np.array([60., 0.])) / np.array([25., 1800.])
    discrepancy = a.residual_sd * np.sqrt(2 / len(params["coefficients"])) * (
        np.cos(xn @ params["frequencies"].T + params["phases"]) @ params["coefficients"]
    )
    # Clipping is part of the synthetic reference definition, not posterior processing.
    return np.clip(base + discrepancy, 0, 1), (lower, upper, co2_limit)


def simulation(seed, scenario, a):
    # Draw all exogenous randomness before computing scenario-specific comfort.
    # Equal seed = equal environment, error draws, reporting uniforms and persona.
    rng = np.random.default_rng(seed)
    params = dict(center=rng.uniform(72, 75), width=rng.uniform(3, 5),
                  co2_limit=rng.uniform(750, 1100),
                  frequencies=rng.normal(0, 5, size=(8, 2)),
                  phases=rng.uniform(0, 2*np.pi, size=8), coefficients=rng.normal(size=8))
    X = rng.uniform([66., 400.], [80., 1500.], size=(a.days, 2))
    errors, report_u = rng.normal(0, a.observation_sd, a.days), rng.uniform(size=a.days)
    if scenario == "annual":
        # Independent stream preserves all pre-existing scenarios and paired
        # environmental/error/reporting draws at the same seed and duration.
        phase_rng = np.random.default_rng(np.random.SeedSequence([seed, 365, 824]))
        params["annual_phase"] = float(phase_rng.uniform(0, 2*np.pi)) if annual_shape(a) == "sinusoidal" else 0.0
    latent = np.array([reference(X[d:d+1], d, scenario, params, a)[0][0] for d in range(a.days)])
    y = np.clip(latent + errors, 0, 1)
    probabilities = np.full(a.days, a.feedback_probability)
    if a.reporting == "complaint":
        # Selection depends on latent comfort, not the particular noisy error draw.
        probabilities = np.clip(a.feedback_probability * (0.25 + 1.5*(1-latent)), 0, 1)
    mask = report_u < probabilities
    return params, X, latent, y, mask


def fit(X, y, ages, tau, amplitude, a):
    """Same model classes/noise rule as production, explicit initialization.

    Always use a FixedNoiseGaussianLikelihood, including no-forgetting. This avoids
    changing the likelihood family or number of learned parameters in the ablation.
    Cold-start each checkpoint, identical initialization and optimization budget.
    """
    baseline = ComfortBaseline({"temp": 1., "co2": 1.}, {k: v.copy() for k, v in PRIOR.items()}, learn_weights=False)
    fixed = torch.tensor(noise_variance(ages, tau, a.base_variance, amplitude), dtype=torch.float32)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=fixed, learn_additional_noise=True,
        noise_constraint=gpytorch.constraints.Interval(a.global_noise_low, a.global_noise_high),
    )
    # Set the intended value after installing its constraint. Production's fitter
    # changes the constraint after assignment, changing the effective initial value.
    likelihood.second_noise = a.global_noise_init
    model = ResidualGP(X, y, likelihood, baseline)
    model.covar_module.outputscale = 0.005
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(a.steps):
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss. Inspect saved inputs; reduce learning rate before rerunning.")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite training gradient; experiment stopped rather than discarding a failed run.")
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood, baseline, float(loss.detach())


def prediction_metrics(mean, variance, target):
    if not all(np.isfinite(v).all() for v in [mean, variance, target]):
        raise RuntimeError("Nonfinite predictions or reference.")
    variance = np.maximum(variance, 1e-10)
    error = mean - target
    radius = Z95 * np.sqrt(variance)
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                nlpd=float(np.mean(0.5*np.log(2*np.pi*variance)+error**2/(2*variance))),
                coverage95=float(np.mean(np.abs(error) <= radius)),
                width95=float(np.mean(2*radius)),
                mean_outside_01=float(np.mean((mean < 0) | (mean > 1))))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def time_average(rows, metric):
    if not rows:
        return None
    times = np.array([r["day"] for r in rows])
    values = np.array([r[metric] for r in rows])
    if len(rows) == 1:
        return float(values[0])
    return float(np.sum(np.diff(times)*(values[:-1]+values[1:])/2)/(times[-1]-times[0]))


def recovery(rows, end, threshold, consecutive):
    after = [r for r in rows if r["day"] >= end]
    for i in range(len(after)-consecutive+1):
        if all(r["rmse"] <= threshold for r in after[i:i+consecutive]):
            return after[i]["day"]-end
    return None


def annual_tracking(rows, harmonic=True):
    """Diagnostics of fitted IEW temperature center, not the full GP surface.

    Fit intercept + sin/cos at the known annual frequency. Positive phase lag
    means the learned cycle trails the true one, modulo one year. Do not report
    phase from a short run or an effectively flat/unexplained learned cycle.
    """
    result = dict(tracking_center_rmse_f=None, tracking_phase_lag_days=None,
                  tracking_amplitude_ratio=None, tracking_harmonic_r2=None)
    if not rows:
        return result
    t = np.array([r["day"] for r in rows], dtype=float)
    truth = np.array([(r["true_temp_lower"]+r["true_temp_upper"])/2 for r in rows])
    learned = np.array([(r["learned_temp_lower"]+r["learned_temp_upper"])/2 for r in rows])
    squared = [dict(day=day, error=float(error)) for day, error in zip(t, (learned-truth)**2)]
    result["tracking_center_rmse_f"] = math.sqrt(time_average(squared, "error"))
    if not harmonic or len(rows) < 8 or t[-1]-t[0] < ANNUAL_PERIOD_DAYS:
        return result
    angle = 2*np.pi*t/ANNUAL_PERIOD_DAYS
    design = np.column_stack([np.ones_like(t), np.sin(angle), np.cos(angle)])
    # Trapezoidal weights prevent extra checkpoints from dominating the fit.
    weights = np.empty_like(t)
    weights[0], weights[-1] = (t[1]-t[0])/2, (t[-1]-t[-2])/2
    weights[1:-1] = (t[2:]-t[:-2])/2
    root = np.sqrt(weights)
    actual_coef = np.linalg.lstsq(design*root[:,None], truth*root, rcond=None)[0]
    fit_coef = np.linalg.lstsq(design*root[:,None], learned*root, rcond=None)[0]
    actual_amp, fit_amp = np.linalg.norm(actual_coef[1:]), np.linalg.norm(fit_coef[1:])
    if actual_amp < 1e-6:
        return result
    result["tracking_amplitude_ratio"] = float(fit_amp/actual_amp)
    ss_total = np.sum(weights*(learned-np.average(learned, weights=weights))**2)
    r2 = 1-np.sum(weights*(learned-design@fit_coef)**2)/ss_total if ss_total > 1e-12 else 0.0
    result["tracking_harmonic_r2"] = float(r2)
    if fit_amp >= 0.05*actual_amp and r2 >= 0.1:
        actual_phase = np.arctan2(actual_coef[2], actual_coef[1])
        fitted_phase = np.arctan2(fit_coef[2], fit_coef[1])
        phase_difference = (actual_phase-fitted_phase+np.pi) % (2*np.pi)-np.pi
        result["tracking_phase_lag_days"] = float(phase_difference*ANNUAL_PERIOD_DAYS/(2*np.pi))
    return result


def summarize(rows, a):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["scenario"], row["variant"]), []).append(row)
    result = []
    for (seed, scenario, name), group in groups.items():
        group.sort(key=lambda r: r["day"])
        annual = scenario == "annual"
        start = a.annual_warmup_days if annual else a.change_day
        end = None if annual else a.change_day + (a.drift_duration if scenario == "gradual" else 0)
        post = [r for r in group if r["day"] >= start]
        settled = [] if annual else [r for r in group if r["day"] >= end]
        lag = None if scenario in ["stable", "annual"] else recovery(group, end, a.recovery_rmse, a.recovery_checkpoints)
        row = dict(seed=seed, scenario=scenario, variant=name, tau_days=group[0]["tau_days"],
                   history_variance=group[0]["history_variance"], n_feedback=group[-1]["n_feedback"],
                   n_evaluations=len(group), first_evaluation_day=group[0]["day"],
                   analysis_start_day=start, analysis_window="after annual warm-up" if annual else "after change onset",
                   mean_rmse=time_average(group, "rmse"), post_rmse=time_average(post, "rmse"),
                   settled_rmse=time_average(settled, "rmse"), final_rmse=group[-1]["rmse"],
                   post_mae=time_average(post, "mae"), post_nlpd=time_average(post, "nlpd"),
                   post_coverage95=time_average(post, "coverage95"), post_width95=time_average(post, "width95"),
                   post_prior_rmse=time_average(post, "prior_rmse"),
                   recovery_days=lag, recovered=None if scenario in ["stable", "annual"] else int(lag is not None),
                   recovery_followup_days=None if annual else a.days-1-end,
                   **annual_tracking(post if annual else [], harmonic=annual_shape(a) == "sinusoidal"),
                   global_noise_bound_hit_fraction=float(np.mean([r["global_noise_near_bound"] for r in group])))
        result.append(row)
    return result


def paired_summary(summaries):
    ref = {(r["seed"], r["scenario"]): r for r in summaries if r["variant"] == "no_forgetting"}
    groups = {}
    for row in summaries:
        groups.setdefault((row["scenario"], row["variant"]), []).append(row)
    output = []
    rng = np.random.default_rng(4921)
    for (scenario, name), group in groups.items():
        delta = np.array([r["post_rmse"]-ref[r["seed"], scenario]["post_rmse"] for r in group])
        interval = [None, None]
        if len(delta) >= 2:
            samples = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
        recovery_times = [r["recovery_days"] for r in group if r["recovered"] == 1]
        output.append(dict(scenario=scenario, variant=name, runs=len(group),
                           mean_post_rmse=float(np.mean([r["post_rmse"] for r in group])),
                           delta_post_rmse_vs_no_forgetting=float(delta.mean()),
                           delta_bootstrap95_low=interval[0], delta_bootstrap95_high=interval[1],
                           fraction_better=float(np.mean(delta < 0)),
                           mean_post_coverage95=float(np.mean([r["post_coverage95"] for r in group])),
                           mean_post_width95=float(np.mean([r["post_width95"] for r in group])),
                           analysis_start_day=group[0]["analysis_start_day"], analysis_window=group[0]["analysis_window"],
                           mean_tracking_center_rmse_f=float(np.mean([r["tracking_center_rmse_f"] for r in group])) if scenario == "annual" else None,
                           recovery_fraction=None if scenario in ["stable", "annual"] else float(np.mean([r["recovered"] for r in group])),
                           median_recovery_days_among_recovered=float(np.median(recovery_times)) if recovery_times else None))
    return output


def plots(rows, summaries, comparisons, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14,
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = [v[0] for v in variants(a)]
    colors = {n: plt.get_cmap("tab10")(i % 10) for i, n in enumerate(names)}
    # A single amplitude belongs in the caption; distinguish it only for sweeps.
    show_amplitude = len(a.history_variances) > 1
    labels = {n: ("No forgetting" if math.isinf(t) else
                  f"tau={t:g} d" + (f", a={h:g}" if show_amplitude else ""))
              for n, t, h in variants(a)}

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    def label_scenario(ax, scenario):
        caption = {"stable": "Stable", "gradual": "Gradual change",
                   "abrupt": "Abrupt change", "annual": "Annual cycle"}[scenario]
        ax.text(0.025, 0.975, caption, transform=ax.transAxes,
                ha="left", va="top", fontsize=14, zorder=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})

    fig, axes = plt.subplots(1, len(a.scenarios), figsize=(5*len(a.scenarios), 3.4), squeeze=False, sharey=True)
    for ax, scenario in zip(axes[0], a.scenarios):
        for name in names:
            group = [r for r in rows if r["scenario"] == scenario and r["variant"] == name]
            days = sorted(set(r["day"] for r in group))
            values = [np.array([r["rmse"] for r in group if r["day"] == day]) for day in days]
            means = np.array([v.mean() for v in values])
            sds = np.array([v.std(ddof=1) if len(v)>1 else 0 for v in values])
            ax.plot(days, means, color=colors[name], label=labels[name])
            ax.fill_between(days, np.maximum(0, means-sds), means+sds, color=colors[name], alpha=0.1)
        if scenario == "abrupt":
            ax.axvline(a.change_day, color="black", ls=":", lw=1)
        elif scenario == "gradual":
            ax.axvspan(a.change_day, a.change_day+a.drift_duration, color="gray", alpha=0.12)
        elif scenario == "annual":
            ax.axvspan(0, a.annual_warmup_days, color="gray", alpha=0.12)
        ax.set(xlabel="Simulation day")
        label_scenario(ax, scenario)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("RMSE against current\nlatent reference")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "learning_curves")

    fig, axes = plt.subplots(1, len(a.scenarios), figsize=(5*len(a.scenarios), max(3.4, len(names)*0.45)), squeeze=False, sharey=True, sharex=True)
    for ax, scenario in zip(axes[0], a.scenarios):
        for i, name in enumerate(names[1:]):
            r = next(c for c in comparisons if c["scenario"] == scenario and c["variant"] == name)
            val, lo, hi = r["delta_post_rmse_vs_no_forgetting"], r["delta_bootstrap95_low"], r["delta_bootstrap95_high"]
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=colors[name], lw=2)
            ax.scatter([val], [i], color=colors[name])
        ax.axvline(0, color="black", ls=":")
        period_label = "after warm-up" if scenario == "annual" else "post-period"
        ax.set(xlabel=f"Paired RMSE difference ({period_label})\nNegative favors forgetting")
        ax.set_yticks(range(len(names)-1), [labels[n] for n in names[1:]])
        ax.set_ylim(-0.6, len(names)-1-0.4)
        label_scenario(ax, scenario)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, "paired_rmse")

    fig, axes = plt.subplots(1, len(a.scenarios), figsize=(5*len(a.scenarios), 3.4), squeeze=False, sharey=True)
    for ax, scenario in zip(axes[0], a.scenarios):
        for name in names:
            g = [r for r in summaries if r["scenario"] == scenario and r["variant"] == name]
            x, y = [r["post_coverage95"] for r in g], [r["post_width95"] for r in g]
            ax.scatter(x, y, color=colors[name], alpha=0.25, s=15)
            ax.scatter(np.mean(x), np.mean(y), color=colors[name], s=65, label=labels[name])
        ax.axvline(0.95, color="black", ls=":", label="Nominal 95% coverage")
        ax.set(xlabel="Empirical latent 95% interval coverage", xlim=(0, 1.02))
        label_scenario(ax, scenario)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean latent interval width")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(0, 0.87))
    fig.tight_layout()
    save(fig, "coverage_width")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ages = np.linspace(0, a.days, 200)
    for name, tau, amplitude in variants(a):
        ax.plot(ages, noise_variance(ages, tau, a.base_variance, amplitude), color=colors[name], label=labels[name])
    ax.set(xlabel="Observation age (days)", ylabel="Fixed observation variance")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.10), labelspacing=0.3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "noise_by_age")

    annual_rows = [r for r in rows if r["scenario"] == "annual"]
    if annual_rows:
        # Show an explicitly labeled example, plus aggregate tracking error.
        # Sinusoidal runs have different phases and should not be averaged raw.
        example_seed = a.seeds[0]
        example = [r for r in annual_rows if r["seed"] == example_seed]
        persona = json.loads((out / "simulation_inputs" / f"seed_{example_seed}_annual_persona.json").read_text())
        step_mode = annual_shape(a) == "seasonal_steps"
        dense_days = np.arange(a.days) if step_mode else np.linspace(0, a.days-1, 1000)
        true_center = persona["center"] + annual_offset(dense_days, persona, a)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(dense_days, true_center, color="black", ls="--", lw=2, label="Reference center",
                     drawstyle="steps-post" if step_mode else "default")
        for name in names:
            g = sorted([r for r in example if r["variant"] == name], key=lambda r: r["day"])
            axes[0].plot([r["day"] for r in g], [(r["learned_temp_lower"]+r["learned_temp_upper"])/2 for r in g],
                         color=colors[name], label=labels[name])
            g = [r for r in annual_rows if r["variant"] == name]
            days = sorted(set(r["day"] for r in g))
            errors = [np.array([abs((r["learned_temp_lower"]+r["learned_temp_upper"]-r["true_temp_lower"]-r["true_temp_upper"])/2)
                                for r in g if r["day"] == day]) for day in days]
            axes[1].plot(days, [np.sqrt(np.mean(e**2)) for e in errors], color=colors[name], label=labels[name])
        for ax in axes:
            ax.axvspan(0, a.annual_warmup_days, color="gray", alpha=0.12)
            ax.grid(alpha=0.2)
        if step_mode:
            for year_start in range(0, a.days, ANNUAL_PERIOD_DAYS):
                for i, start in enumerate(SEASON_STARTS):
                    left = year_start+start
                    right = min(year_start+(SEASON_STARTS[i+1] if i < 3 else ANNUAL_PERIOD_DAYS), a.days-1)
                    if left >= right:
                        continue
                    axes[0].text((left+right)/2, .98, SEASON_NAMES[i], ha="center", va="top",
                                 fontsize=8, transform=axes[0].get_xaxis_transform())
                    for ax in axes:
                        ax.axvline(left, color="gray", alpha=.25, lw=.6)
            axes[0].margins(y=.18)
        axes[0].set(ylabel="Temperature center (F)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=8)
        axes[1].set(ylabel="Center RMSE across runs (F)", xlabel="Simulation day")
        fig.tight_layout(rect=(0, .08, 1, .97))
        save(fig, "annual_preference_tracking")


def evaluation_days(scenario, a):
    days = set(range(a.eval_every, a.days, a.eval_every)) | {a.days-1}
    if scenario == "annual":
        days.add(a.annual_warmup_days)
        days.update(range(ANNUAL_PERIOD_DAYS, a.days, ANNUAL_PERIOD_DAYS))
        if annual_shape(a) == "seasonal_steps":
            # Observe both sides of each jump, avoiding a month-long interpolated
            # ramp in the plotted errors and their trapezoidal time averages.
            for year_start in range(0, a.days, ANNUAL_PERIOD_DAYS):
                for start in SEASON_STARTS:
                    boundary = year_start+start
                    if 0 < boundary < a.days:
                        days.update([boundary-1, boundary])
    else:
        days.update([a.change_day-1, a.change_day, a.change_day+a.drift_duration])
    return sorted(days)


def run(a, out):
    torch.set_num_threads(a.threads)
    (out / 'simulation_inputs').mkdir(parents=True, exist_ok=True)
    grid = np.stack(np.meshgrid(np.linspace(66, 80, a.grid_size), np.linspace(400, 1500, a.grid_size)), axis=-1).reshape(-1, 2)
    Xt = torch.tensor(grid, dtype=torch.float32)
    prior = ComfortBaseline({'temp': 1.0, 'co2': 1.0}, PRIOR, learn_weights=False)
    with torch.no_grad():
        prior_mean = prior(Xt).numpy()
    all_rows = []
    for seed in a.seeds:
        for scenario in a.scenarios:
            eval_days = evaluation_days(scenario, a)
            params, X, latent, y, mask = simulation(seed, scenario, a)
            days = np.arange(a.days)[mask]
            np.savez_compressed(out / 'simulation_inputs' / f'seed_{seed}_{scenario}.npz', days=np.arange(a.days), X=X, latent=latent, observed=y, reported=mask, evaluation_grid=grid)
            analysis_start = a.annual_warmup_days if scenario == 'annual' else a.change_day - 1
            if (days <= analysis_start).sum() < 3:
                raise RuntimeError(f'Seed {seed}: fewer than 3 observations by analysis start (day {analysis_start}). Increase warm-up/duration/rate; do not silently drop this run.')
            targets = {day: reference(grid, day, scenario, params, a) for day in eval_days}
            for name, tau, amplitude in variants(a):
                run_rows = []
                for day in eval_days:
                    selected = mask & (np.arange(a.days) <= day)
                    if selected.sum() < 3:
                        print(f'seed={seed} {scenario} {name} day={day}: not evaluated (fewer than 3 observations)', flush=True)
                        continue
                    torch.manual_seed(seed)
                    model, likelihood, baseline, loss = fit(torch.tensor(X[selected], dtype=torch.float32), torch.tensor(y[selected], dtype=torch.float32), day - np.arange(a.days)[selected], tau, amplitude, a)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
                        pred = model(Xt)
                        mean, var = (pred.mean.numpy(), pred.variance.numpy())
                        low, high = baseline.get_comfort_bounds('temp')
                        _, co2 = baseline.get_comfort_bounds('co2')
                    target, true_bounds = targets[day]
                    global_noise = float(likelihood.second_noise.detach())
                    near = min(global_noise - a.global_noise_low, a.global_noise_high - global_noise) <= 0.01 * (a.global_noise_high - a.global_noise_low)
                    row = dict(seed=seed, scenario=scenario, variant=name, tau_days='inf' if math.isinf(tau) else tau, history_variance=amplitude, day=day, n_feedback=int(selected.sum()), **prediction_metrics(mean, var, target), prior_rmse=float(np.sqrt(np.mean((prior_mean - target) ** 2))), learned_temp_lower=float(low), learned_temp_upper=float(high), learned_co2_upper=float(co2), true_temp_lower=true_bounds[0], true_temp_upper=true_bounds[1], true_co2_upper=true_bounds[2], global_noise=global_noise, global_noise_near_bound=int(near), final_training_loss=loss)
                    run_rows.append(row)
                    print(f"seed={seed} {scenario} {name} day={day} n={selected.sum()} RMSE={row['rmse']:.4f}", flush=True)
                all_rows.extend(run_rows)
                write_csv(out / 'metrics_by_checkpoint.csv', all_rows)
    summaries = summarize(all_rows, a)
    comparisons = paired_summary(summaries)
    write_csv(out / 'metrics_by_run.csv', summaries)
    write_csv(out / 'paired_comparisons.csv', comparisons)
    plots(all_rows, summaries, comparisons, a, out)
    print(f'\nCompleted. Results: {out.resolve()}')
    if a.quick:
        print('Quick-run figures only verify execution; they are not evidence for the manuscript.')
