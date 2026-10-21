import math
import numpy as np
import torch
import gpytorch
from . import temporal
METRICS = ("rmse", "mae", "nlpd", "coverage95", "width95")

def corrupt_feedback(y, mask, a):
    days = np.arange(len(y))
    candidates = np.flatnonzero(mask & (days >= a.attack_start)
                               & (days < a.attack_start + a.attack_window))
    if len(candidates) < a.attack_count:
        raise ValueError(f"Only {len(candidates)} existing reports in the attack window; need {a.attack_count}. "
                         "Increase the window or reduce attack-count explicitly; no seed or observation was silently replaced.")
    attack_days = candidates[:a.attack_count]
    corrupted = y.copy()
    corrupted[attack_days] = (np.where(y[attack_days] >= .5, 0., 1.)
                              if a.error_mode == "opposite_endpoint" else 1. - y[attack_days])
    other = np.ones(len(y), dtype=bool)
    other[attack_days] = False
    np.testing.assert_array_equal(corrupted[other], y[other])
    return corrupted, attack_days


def prepare(a):
    grid = np.stack(np.meshgrid(np.linspace(66, 80, a.grid_size),
                               np.linspace(400, 1500, a.grid_size)), axis=-1).reshape(-1, 2)
    runs = {}
    audits = []
    for seed in a.seeds:
        params, X, latent, unmodified, mask = temporal.simulation(seed, "stable", a)
        corrupted, attack_days = corrupt_feedback(unmodified, mask, a)
        reported = np.flatnonzero(mask)
        if (reported < attack_days[0]).sum() < 3:
            raise ValueError(f"Seed {seed}: fewer than three reports before attack; increase warm-up.")
        target = temporal.reference(grid, 0, "stable", params, a)[0]
        np.testing.assert_array_equal(target, temporal.reference(grid, a.days - 1, "stable", params, a)[0])
        runs[seed] = dict(params=params, X=X, latent=latent, unmodified=unmodified,
                          corrupted=corrupted, mask=mask, attack_days=attack_days,
                          target=target, third_report_day=int(reported[2]))
        for day in attack_days:
            audits.append(dict(seed=seed, day=int(day), feedback_number=int(mask[:day+1].sum()),
                               temperature_F=float(X[day, 0]), co2_ppm=float(X[day, 1]),
                               latent_comfort=float(latent[day]), unmodified_score=float(unmodified[day]),
                               corrupted_score=float(corrupted[day]),
                               score_change=float(corrupted[day] - unmodified[day]),
                               error_from_latent=float(corrupted[day] - latent[day])))
    days = set(range(a.eval_every, a.days, a.eval_every))
    days.update(range(a.attack_start, min(a.days, a.attack_start + a.attack_window + 61), a.event_eval_every))
    days.update([a.attack_start - 1, a.attack_start + a.attack_window - 1, a.days - 1])
    # Every plotted time point contains the same seeds; early insufficient prefixes
    # are excluded for all runs, never selectively omitted from an average.
    warmup = max(r["third_report_day"] for r in runs.values())
    eval_days = sorted(day for day in days if day >= warmup)
    return grid, runs, audits, eval_days


def variants(a):
    return [("no_forgetting", math.inf), *[(f"tau_{tau:g}", tau) for tau in a.taus]]


def evaluate_fit(seed, variant, tau, day, branch, run, grid, a):
    calendar = np.arange(a.days)
    selected = run["mask"] & (calendar <= day)
    torch.manual_seed(seed)
    model, likelihood, _, loss = temporal.fit(
        torch.tensor(run["X"][selected], dtype=torch.float32),
        torch.tensor(run[branch][selected], dtype=torch.float32),
        day - calendar[selected], tau, a.history_variances[0], a)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        latent = model(torch.tensor(grid, dtype=torch.float32))
        metrics = temporal.prediction_metrics(latent.mean.numpy(), latent.variance.numpy(), run["target"])
    return dict(seed=seed, variant=variant, tau_days="inf" if math.isinf(tau) else tau,
                day=day, branch=branch, n_feedback=int(selected.sum()),
                n_corrupted_available=int((run["attack_days"] <= day).sum()) if branch == "corrupted" else 0,
                **metrics, global_noise=float(likelihood.second_noise.detach()), training_loss=loss)


def paired_rows(rows):
    lookup = {(r["seed"], r["variant"], r["day"], r["branch"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["branch"] != "corrupted":
            continue
        clean = lookup[row["seed"], row["variant"], row["day"], "unmodified"]
        pairs.append(dict(seed=row["seed"], variant=row["variant"], day=row["day"],
                          **{f"delta_{k}": row[k] - clean[k] for k in METRICS}))
    return pairs


def summarize(pairs, runs, a):
    result = []
    for seed in a.seeds:
        for variant, _ in variants(a):
            group = sorted([r for r in pairs if r["seed"] == seed and r["variant"] == variant], key=lambda r:r["day"])
            first, last = map(int, (runs[seed]["attack_days"][0], runs[seed]["attack_days"][-1]))
            affected = [r for r in group if r["day"] >= first]
            after = [r for r in group if r["day"] > last]
            recovered = None
            for i in range(len(after) - a.recovery_checkpoints + 1):
                window = after[i:i+a.recovery_checkpoints]
                if all(r["delta_rmse"] <= a.recovery_tolerance for r in window):
                    recovered = window[0]["day"] - last
                    break
            peak = max(affected, key=lambda r:r["delta_rmse"])
            result.append(dict(seed=seed, variant=variant, first_error_day=first, last_error_day=last,
                               peak_excess_rmse=peak["delta_rmse"], peak_day=peak["day"],
                               time_averaged_excess_rmse=temporal.time_average(affected, "delta_rmse"),
                               final_excess_rmse=group[-1]["delta_rmse"],
                               recovery_observed=int(recovered is not None), recovery_days=recovered))
    return result


def plots(rows, pairs, summaries, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"],
                         "font.size":12, "pdf.fonttype":42, "svg.fonttype":"none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    colors = plt.get_cmap("tab10")
    for i, (name, tau) in enumerate(variants(a)):
        label = "No forgetting" if math.isinf(tau) else rf"$\tau$ = {tau:g} d"
        for ax, branch in zip(axes[:2], ["unmodified", "corrupted"]):
            group = [r for r in rows if r["variant"] == name and r["branch"] == branch]
            days = sorted({r["day"] for r in group})
            values = np.array([[next(r["rmse"] for r in group if r["seed"] == seed and r["day"] == day)
                                for day in days] for seed in a.seeds])
            mean = values.mean(0)
            sd = values.std(0, ddof=1) if len(a.seeds) > 1 else np.zeros_like(mean)
            ax.plot(days, mean, color=colors(i), label=label, lw=1.7)
            ax.fill_between(days, np.maximum(0, mean-sd), mean+sd, color=colors(i), alpha=.10)
        group = [r for r in pairs if r["variant"] == name]
        days = sorted({r["day"] for r in group})
        values = np.array([[next(r["delta_rmse"] for r in group if r["seed"] == seed and r["day"] == day)
                            for day in days] for seed in a.seeds])
        mean = values.mean(0)
        sd = values.std(0, ddof=1) if len(a.seeds) > 1 else np.zeros_like(mean)
        axes[2].plot(days, mean, color=colors(i), lw=1.7)
        axes[2].fill_between(days, mean-sd, mean+sd, color=colors(i), alpha=.10)
    ymax = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(0, ymax)
        ax.set_ylabel("RMSE against latent reference")
    axes[2].axhline(0, color="0.3", ls="--", lw=1)
    axes[2].set_ylabel("RMSE increase from score errors")
    for ax, title in zip(axes, ["(a) Unmodified feedback", "(b) Severe-error burst", "(c) Paired RMSE difference"]):
        ax.set(title=title, xlabel="Simulation day", xlim=(0, a.days-1))
        ax.axvspan(a.attack_start, a.attack_start+a.attack_window, color="0.4", alpha=.10)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=True, framealpha=.9,
                   facecolor="white", edgecolor="0.8", fontsize=11)
    if a.quick:
        fig.suptitle("SMOKE TEST ONLY: reduced fitting budget", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93 if a.quick else 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"adversarial_learning_curves.{ext}", dpi=300)
    plt.close(fig)

    lines = ["# Severe interpretation-error sensitivity", "",
             "SMOKE TEST ONLY; not manuscript evidence." if a.quick else "Full configured run.", "",
             f"Fixed preferences; {len(a.seeds)} seeds; first {a.attack_count} existing reports in days "
             f"[{a.attack_start}, {a.attack_start+a.attack_window}) are corrupted ({a.error_mode}).",
             "Shaded vertical region is the eligible injection window, not continuous corruption.",
             "Colored ribbons are mean ± one sample SD across simulation seeds, not confidence intervals.",
             "Paired differences are corrupted minus unmodified RMSE, computed within seed and setting.",
             "Positive differences indicate harm; negative differences indicate an incidental improvement.",
             "Unmodified feedback already includes Gaussian observation noise and clipping.",
             "No LLM is invoked; these are deliberately corrupted numerical scores, not validated language attacks.",
             "Fitting uses the Section 3.4 cold-start checkpoint protocol, not warm-start online optimization.",
             "Evaluation is against the fixed latent baseline plus smooth discrepancy, without observation noise.",
             "", "## Summary across seeds", "",
             "| Setting | Mean peak excess RMSE | Mean time-averaged excess | Mean final excess | Recovery criterion met |",
             "|---|---:|---:|---:|---:|"]
    for name, _ in variants(a):
        group = [r for r in summaries if r["variant"] == name]
        lines.append(f"| {name} | {np.mean([r['peak_excess_rmse'] for r in group]):.4f} | "
                     f"{np.mean([r['time_averaged_excess_rmse'] for r in group]):.4f} | "
                     f"{np.mean([r['final_excess_rmse'] for r in group]):.4f} | "
                     f"{sum(r['recovery_observed'] for r in group)}/{len(group)} |")
    lines.extend(["", f"Recovery means excess RMSE <= {a.recovery_tolerance:g} for "
                  f"{a.recovery_checkpoints} consecutive evaluated checkpoints after the last corrupted feedback. "
                  "Delay is measured from that last feedback to the first qualifying checkpoint. "
                  "This is a chosen descriptive threshold, not an established recovery standard, and does not guarantee no later rebound. "
                  "Peak errors and recovery times are measured at the scheduled evaluation checkpoints. "
                  "Missing recovery times are censored at the end of the simulation, not set to zero."])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "caption.tex").write_text(
        r"\caption{Sensitivity to a short burst of severe numerical feedback errors under fixed synthetic comfort preferences. "
        r"(a) RMSE with unmodified feedback, (b) RMSE with corrupted feedback, and (c) their within-seed paired difference. "
        f"The first {a.attack_count} existing feedback events within the shaded eligibility window are corrupted; "
        r"all other scores, environmental inputs, and reporting times are unchanged. "
        r"Lines show means and ribbons show $\pm$ one standard deviation across simulation seeds. "
        r"Positive paired differences indicate increased error. This numerical sensitivity test does not evaluate adversarial language or actual LLM interpretation errors.}" + "\n")
