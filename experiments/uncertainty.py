import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
BLUE, RED, ORANGE = "#0072B2", "#C43C39", "#D55E00"

def coverage_metrics(mean, variance, reference, nominal):
    mean, variance, reference = [np.asarray(a, dtype=np.float64)
                                 for a in (mean, variance, reference)]
    if mean.ndim != 1 or mean.shape != variance.shape or mean.shape != reference.shape:
        raise ValueError("Expected matching one-dimensional mean, variance, and reference arrays.")
    if not len(mean) or not all(np.isfinite(a).all() for a in (mean, variance, reference)):
        raise ValueError("Empty or nonfinite evaluation arrays.")
    if np.any(variance <= 0) or not 0 < nominal < 1:
        raise ValueError("Require positive variance and nominal coverage strictly between 0 and 1.")
    half_width = norm.ppf((1 + nominal) / 2) * np.sqrt(variance)
    included = np.abs(reference - mean) <= half_width
    return dict(nominal_coverage=float(nominal), n_conditions=len(mean),
                covered_conditions=int(included.sum()),
                empirical_coverage=float(included.mean()),
                mean_interval_width=float((2 * half_width).mean()),
                mae=float(np.abs(reference - mean).mean()),
                rmse=float(np.sqrt(np.mean((reference - mean) ** 2))),
                nlpd=float(np.mean(0.5 * np.log(2 * np.pi * variance)
                                   + 0.5 * (reference - mean) ** 2 / variance)))


def draw_figure(data, out, width, height):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "axes.titlesize": 11, "legend.fontsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(width, height), sharey="row")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.11, wspace=0.14, hspace=0.40)
    for column, item in enumerate(data):
        top, bottom = axes[:, column]
        x, mean = item["temperature"], item["slice_mean"]
        delta = norm.ppf(.975) * np.sqrt(item["slice_variance"])
        top.fill_between(x, mean - delta, mean + delta, color=BLUE, alpha=.22,
                         label="Nominal 95% latent interval")
        top.plot(x, item["slice_reference"], color=RED, lw=1.7, label="Synthetic IEW reference")
        top.plot(x, mean, color=BLUE, lw=1.7, label="Posterior mean")
        top.set(title=f"({chr(97 + column)}) {item['persona']}: CO$_2$ = {item['co2']:g} ppm",
                xlabel="Temperature (°F)", xlim=(x.min(), x.max()))
        top.set_xticks(np.arange(5 * np.ceil(x.min()/5), x.max() + .01, 5))
        for target, color, marker, label in [("iew", BLUE, "o", "IEW reference"),
                                              ("iew_plus_residual", ORANGE, "s", "IEW + sampled residual")]:
            subset = [r for r in item["rows"] if r["target"] == target]
            bottom.plot([r["nominal_coverage"] * 100 for r in subset],
                        [r["empirical_coverage"] * 100 for r in subset],
                        color=color, marker=marker, markersize=3.8, lw=1.5, label=label,
                        clip_on=False)
        bottom.plot([0, 100], [0, 100], "--", color="0.4", lw=1, label="Nominal = empirical")
        bottom.axvline(95, color="0.6", ls=":", lw=.8)
        bottom.set(title=f"({chr(100 + column)}) {item['persona']}: full evaluation grid",
                   xlabel="Nominal coverage (%)", xlim=(30, 105), ylim=(30, 105),
                   xticks=[30, 50, 70, 90, 100], yticks=[30, 50, 70, 90, 100])
        at95 = {r["target"]: r for r in item["rows"] if np.isclose(r["nominal_coverage"], .95)}
        bottom.text(.03, .97,
                    f"95% coverage: IEW {100*at95['iew']['empirical_coverage']:.1f}%\n"
                    f"IEW + residual {100*at95['iew_plus_residual']['empirical_coverage']:.1f}%\n"
                    f"Mean 95% width: {at95['iew']['mean_interval_width']:.3f}",
                    ha="left", va="top", fontsize=9, transform=bottom.transAxes,
                    bbox=dict(facecolor="white", alpha=.92, edgecolor="none", pad=2))
        for ax in (top, bottom):
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set(ylabel="Comfort score", ylim=(0, 1.01))
    axes[1, 0].set_ylabel("Empirical coverage (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend([handles[i] for i in [1, 2, 0]], [labels[i] for i in [1, 2, 0]],
                      loc="lower center", fontsize=9, framealpha=.95,
                      borderpad=.3, labelspacing=.2, handlelength=1.8)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="lower right", fontsize=9, framealpha=.95,
                      borderpad=.3, labelspacing=.2, handlelength=1.8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"uncertainty_calibration.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
