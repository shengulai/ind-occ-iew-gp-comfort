import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
COLORS = {"llm_checkpoint":"#0072B2","sampled_numerical":"#D55E00","reference":"#333333"}
ARM_LABELS = {"llm_checkpoint":"LLM-interpreted","sampled_numerical":"Sampled numerical"}
BOUND_KEYS=("temp_lower","temp_upper","co2_lower","co2_upper")

def draw_panels(data, axes, panel_labels=('a', 'b'), legends=False):
    counts, reference = data['counts'], data['reference']
    for arm, key in [('llm_checkpoint', 'llm_bounds'), ('sampled_numerical', 'numerical_bounds')]:
        b = data[key]
        color = COLORS[arm]
        for index, style in [(0, '-'), (1, '--')]:
            axes[0].plot(counts, b[:, index], color=color, ls=style, lw=1.25)
        axes[1].plot(counts, b[:, 3], color=color, lw=1.25)
        invalid = b[:, 3] <= b[:, 2]
        if invalid.any():
            axes[1].scatter(counts[invalid], b[invalid, 3], marker='x', color=color, s=25, zorder=5)
    axes[0].axhline(reference[0], color=COLORS['reference'], lw=1.0)
    axes[0].axhline(reference[1], color=COLORS['reference'], ls='--', lw=1.0)
    axes[1].axhline(reference[3], color=COLORS['reference'], ls='--', lw=1.0)
    axes[0].set_title(f"({panel_labels[0]}) {data['persona']}: temperature bounds", loc='left', pad=6)
    axes[1].set_title(f"({panel_labels[1]}) {data['persona']}: CO$_2$ tolerance", loc='left', pad=6)
    axes[0].set_ylabel('Temperature (°F)')
    axes[1].set_ylabel('CO$_2$ upper bound (ppm)')
    for axis in axes:
        axis.set_xlabel('Feedback observations, n')
        axis.set_xlim(-.15, counts[-1] + max(.15, .012*counts[-1]))
        if counts[-1] == 8:
            axis.set_xticks([0, 2, 4, 6, 8])
        elif counts[-1] == 23:
            axis.set_xticks([0, 5, 10, 15, 20, 23])
        else:
            axis.set_xticks([0, 15, 30, 45, 60])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(alpha=.16, linewidth=.5)
        # Reserve headroom for inside-panel legends without hiding the data.
        lo, hi = axis.get_ylim()
        span = max(hi-lo, 1)
        axis.set_ylim(lo - .05*span, hi + (.42 if legends else .07)*span)
    if legends:
        for column, axis in enumerate(axes):
            handles = [Line2D([], [], color=COLORS[arm], lw=1.4, label=ARM_LABELS[arm]) for arm in ARM_LABELS]
            handles.append(Line2D([], [], color=COLORS['reference'], lw=1.0,
                                  ls='-' if column == 0 else '--', label='Synthetic reference'))
            axis.legend(handles=handles, loc='upper left', fontsize=8.5, frameon=True,
                        facecolor='white', edgecolor='none', framealpha=.92, labelspacing=.25,
                        handlelength=2.2, borderpad=.3)
        axes[0].text(.98, .97, 'Solid: lower\nDashed: upper', ha='right', va='top',
                     transform=axes[0].transAxes, fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))


def draw_figures(datasets, out):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    # Match the 11 x 6 inch publication calibration figure: personas in columns.
    fig, axes = plt.subplots(2, len(datasets), figsize=(11, 6), squeeze=False)
    for i, data in enumerate(datasets):
        labels = (chr(97+i), chr(97+len(datasets)+i))
        draw_panels(data, axes[:, i], labels, legends=i == 0)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    for ext in ('png', 'pdf', 'svg'):
        # Preserve the physical size and aspect ratio in every export.
        fig.savefig(out / f'feedback_bound_comparison.{ext}', dpi=300)
    plt.close(fig)
    for data in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
        draw_panels(data, axes, legends=True)
        fig.tight_layout(w_pad=1.5)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / f"{data['persona']}_bound_comparison.{ext}", dpi=300, bbox_inches='tight')
        plt.close(fig)
