import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#222222"
STYLE = {
    "font.family": "Times New Roman", "font.size": 7.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "mathtext.fallback": None,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
}




def snapshot_indices(n, count):
    return np.unique(np.rint(np.linspace(0, n, min(n, count)))).astype(int).clip(1, n) - 1


def rectangle(ax, bounds, color, linestyle, halo=False):
    lo, hi, bottom, top = bounds
    if hi <= lo or top <= bottom:
        return False
    patch = Rectangle((lo, bottom), hi - lo, top - bottom, fill=False,
                      edgecolor=color, linestyle=linestyle, linewidth=1.25, zorder=4)
    if halo:
        patch.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
    ax.add_patch(patch)
    return True


def evolution_axes(axes, data, selected=()):
    n, b, t = data["counts"], data["bounds"], data["truth"]
    temp, co2 = axes
    learned, reference = [], []
    for index, style, label in [(0, "-", "Lower bound"), (1, "--", "Upper bound")]:
        learned.append(temp.plot(n, b[:, index], color=BLUE, ls=style,
                                 label=f"Learned {label.lower()}")[0])
        reference.append(temp.axhline(t[index], color=INK, ls=style, lw=0.8,
                                      label=f"Reference {label.lower()}"))
    interval = temp.fill_between(n, b[:, 0], b[:, 1], color=BLUE, alpha=0.10,
                                 label="Comfort interval (not CI)")
    temp_handles = [learned[0], reference[0], interval, learned[1], reference[1]]
    temp.set_ylabel("Temperature (°F)")
    co2.plot(n, b[:, 3], color=BLUE, label="Learned upper bound")
    co2.axhline(t[3], color=INK, ls="--", lw=0.9, label="Reference upper bound")
    co2.set_ylabel("CO$_2$ upper bound (ppm)")
    invalid = (b[:, 1] <= b[:, 0]) | (b[:, 3] <= b[:, 2])
    if invalid.any():
        co2.scatter(n[invalid], b[invalid, 3], marker="x", s=20, c=ORANGE, zorder=5,
                    label="Invalid interval")
    for ax in axes:
        ax.set_xlabel("Feedback observations, n")
        ax.set_xlim(0.5, n[-1] + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(length=3, width=0.6)
        snapshot = None
        for j, i in enumerate(selected):
            snapshot = ax.axvline(n[i], color="#999999", ls=":", lw=0.6, zorder=0,
                                 label="Snapshot step" if j == len(selected) - 1 else "_nolegend_")
        if ax is temp and snapshot is not None:
            temp_handles.append(snapshot)
        # Reserve headroom for the in-panel legends, keeping them off the data.
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 0.60 * (upper - lower))
    legend_style = dict(loc="upper right", ncol=2, frameon=True, framealpha=0.95,
                        facecolor="white", edgecolor="none", fontsize=6.5,
                        columnspacing=1.1, handlelength=2.1, borderpad=0.4)
    temp.legend(handles=temp_handles, **legend_style)
    co2.legend(**legend_style)


def save_figure(fig, destination, stem, dpi, top_trim=0):
    # Remove unused heading space without changing panel sizes or typography.
    width, height = fig.get_size_inches()
    bounds = Bbox.from_bounds(0, 0, width, height - top_trim)
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"{stem}.{extension}", dpi=dpi, bbox_inches=bounds)
    plt.close(fig)


def plot_persona(data, args, destination):
    selected = snapshot_indices(len(data["counts"]), args.snapshot_count)
    surfaces = "means" in data
    # Keep lower panels, gaps and fonts at their original physical size while
    # changing only the snapshot axes height. The original reference was 5.35 in.
    snapshot_height = args.snapshot_height
    snapshot_bottom = 0.535 * 5.35
    figure_height = 4.95 - 0.34 * 5.35 + snapshot_height
    y_scale = 5.35 / figure_height
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, figure_height))
        top = fig.add_gridspec(1, len(selected), left=0.085, right=0.89 if surfaces else 0.98,
                              bottom=snapshot_bottom / figure_height,
                              top=(snapshot_bottom + snapshot_height) / figure_height, wspace=0.19)
        image = None
        for j, index in enumerate(selected):
            ax = fig.add_subplot(top[0, j])
            ax.set_title(f"{chr(97 + j)}   n = {data['counts'][index]}", loc="left", pad=6)
            if surfaces:
                image = ax.pcolormesh(data["temperature"], data["co2"], data["means"][index],
                                      cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
                points = data["inputs"][:index + 1]
                ax.scatter(points[:-1, 0], points[:-1, 1], s=5, facecolors="white",
                           edgecolors="#444444", linewidths=0.25, alpha=0.6, zorder=3)
                ax.scatter(*points[-1], s=28, marker="*", c="white", edgecolors="black",
                           linewidths=0.6, zorder=5)
            else:
                ax.set_facecolor("#f7f8fa")
            rectangle(ax, data["truth"], INK, "--", halo=surfaces)
            valid = rectangle(ax, data["bounds"][index], ORANGE, "-", halo=surfaces)
            if not valid:
                ax.text(0.5, 0.5, "Invalid learned\ninterval", transform=ax.transAxes,
                        ha="center", color=ORANGE, fontsize=7)
            ax.set(xlim=(60, 85), ylim=(0, 1800), xticks=[60, 70, 80],
                   yticks=[0, 600, 1200, 1800], xlabel="Temperature (°F)")
            if j == 0:
                ax.set_ylabel("CO$_2$ (ppm)")
            else:
                ax.tick_params(labelleft=False)
        if image is not None:
            bar = fig.colorbar(image, cax=fig.add_axes([0.912, snapshot_bottom / figure_height,
                                                      0.013, snapshot_height / figure_height]),
                               ticks=[0, 0.5, 1], extend="both")
            bar.set_label("Predicted comfort score", fontsize=7, labelpad=3)
        handles = [Line2D([], [], c=ORANGE, label="Learned baseline zone"),
                   Line2D([], [], c=INK, ls="--", label="Synthetic reference comfort zone")]
        if surfaces:
            handles.extend([Line2D([], [], ls="", marker="o", color="#888888", ms=3,
                                   label="Previous feedback"),
                            Line2D([], [], ls="", marker="*", color=INK, ms=5,
                                   label="Newest feedback")])
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 0.44 * y_scale),
                   ncol=len(handles), frameon=False, columnspacing=1.4)
        bottom = fig.add_gridspec(1, 2, left=0.085, right=0.975,
                                 bottom=0.09 * y_scale, top=0.355 * y_scale,
                                 wspace=0.25)
        axes = [fig.add_subplot(bottom[0, j]) for j in range(2)]
        evolution_axes(axes, data, selected)
        for j, (ax, title) in enumerate(zip(axes, ["Temperature bounds", "CO$_2$ tolerance"])):
            ax.set_title(f"{chr(97 + len(selected) + j)}   {title}", loc="left")
        save_figure(fig, destination, f"{data['persona']}_learning_evolution", args.dpi)


def plot_comparison(histories, args, destination):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(histories), 2, figsize=(7.2, 1.9 * len(histories) + 0.55),
                                 squeeze=False)
        fig.subplots_adjust(left=0.10, right=0.975, bottom=0.09, top=0.92, hspace=0.65, wspace=0.25)
        for row, data in enumerate(histories):
            evolution_axes(axes[row], data)
            for j, title in enumerate(["Temperature bounds", "CO$_2$ tolerance"]):
                axes[row, j].set_title(f"{data['persona']} | {title}", loc="left")
        # Shared physical scales make between-persona comparisons meaningful.
        for col in [0, 1]:
            columns = [0, 1] if col == 0 else [3]
            values = np.concatenate([np.concatenate((data["bounds"][:, columns].ravel(),
                                                      data["truth"][columns]))
                                     for data in histories])
            span = max(float(np.ptp(values)), 1.0 if col == 0 else 20.0)
            for ax in axes[:, col]:
                ax.set_ylim(values.min() - 0.05 * span, values.max() + 0.70 * span)
        save_figure(fig, destination, "persona_comparison", args.dpi, top_trim=0.05 * fig.get_figheight())
