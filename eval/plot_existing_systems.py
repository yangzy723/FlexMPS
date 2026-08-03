import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap


# ============================================================
# Data
# ============================================================

systems = [
    "Exclusive",
    "MPS",
    "MIG",
    "Salus",
    "Orion",
    "LithOS",
    r"$\mu$Share",
]

operators = [
    "Scatter\nAdd",
    "Conv1D",
    "Linear",
    "LayerNorm",
    "GEMV",
]

# Compress all-zero rows while preserving emphasis on rows with deviations.
row_heights = np.array([0.58, 0.58, 0.58, 0.58, 0.58, 1.18, 1.18])
row_edges = np.concatenate(([0.0], np.cumsum(row_heights)))
row_centers = (row_edges[:-1] + row_edges[1:]) / 2

data_64 = np.array([
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0.0485891, 0.0003266, 0,         0,         0],
    [0.2421278, 0.0008531, 0,         0.0136037, 0.0001795],
])

data_1024 = np.array([
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0,         0,         0,         0,         0],
    [0.0903844, 0.0006317, 0,         0.0001856, 0],
    [0.3131283, 0.0019553, 0,         0.1436051, 0.0002718],
])


# ============================================================
# Paper-style configuration
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    # Preserve editable text in PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# A restrained light-to-deep red palette.
cmap = LinearSegmentedColormap.from_list(
    "paper_red",
    ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
)
cmap.set_bad("#f3f3f3")       # exact zero
cmap.set_under("#f3f3f3")


all_values = np.concatenate((data_64.ravel(), data_1024.ravel()))
nonzero_values = all_values[all_values > 0]

norm = LogNorm(
    vmin=nonzero_values.min(),
    vmax=nonzero_values.max(),
)


# ============================================================
# Formatting helpers
# ============================================================

def format_value(value):
    """Compact cell annotation suitable for a paper figure."""
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    return rf"${mantissa:.2f}\mathrm{{e}}{{{exponent}}}$"


def annotation_color(value):
    """Use white text only on sufficiently dark cells."""
    if value == 0:
        return "#777777"

    normalized = norm(value)
    return "white" if normalized > 0.66 else "black"


def draw_panel(ax, data, title, show_ylabels):
    masked = np.ma.masked_equal(data, 0)

    image = ax.pcolormesh(
        np.arange(len(operators) + 1) - 0.5,
        row_edges,
        masked,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.45,
    )

    ax.set_xlim(-0.5, len(operators) - 0.5)
    ax.set_ylim(row_edges[-1], 0)

    ax.set_title(title, pad=4, fontweight="normal")

    ax.set_xticks(np.arange(len(operators)))
    ax.set_xticklabels(operators)
    ax.tick_params(axis="x", pad=2)

    ax.set_yticks(row_centers)
    if show_ylabels:
        ax.set_yticklabels(systems)
        ax.get_yticklabels()[0].set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)

    # Separate fixed-structure systems from reshaping systems.
    ax.axhline(
        row_edges[5],
        color="#333333",
        linewidth=0.8,
        linestyle=(0, (2.5, 1.5)),
    )

    # Cell annotations.
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]

            ax.text(
                col,
                row_centers[row],
                format_value(value),
                ha="center",
                va="center",
                fontsize=8 if row >= 5 else 6.4,
                color=annotation_color(value),
            )

    # Minimal outer frame.
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.6)

    return image


# ============================================================
# Figure
# ============================================================

# Approximately the width of a two-column paper figure.
fig = plt.figure(figsize=(7.05, 2.05))
grid = fig.add_gridspec(
    1,
    3,
    width_ratios=[1, 1, 0.026],
    left=0.145,
    right=0.905,
    bottom=0.255,
    top=0.865,
    wspace=0.075,
)
axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
]
axes[1].sharey(axes[0])
cbar_ax = fig.add_subplot(grid[0, 2])

image = draw_panel(
    axes[0],
    data_64,
    r"(a) Block size = 64",
    show_ylabels=True,
)

draw_panel(
    axes[1],
    data_1024,
    r"(b) Block size = 1024",
    show_ylabels=False,
)

# Shared colorbar.
cbar = fig.colorbar(
    image,
    cax=cbar_ax,
    orientation="vertical",
)

cbar.set_label(
    "Max Numerical Deviation",
    rotation=90,
    labelpad=5,
)

cbar.ax.tick_params(
    labelsize=6.8,
    width=0.5,
    length=2,
)

cbar.outline.set_linewidth(0.5)

# Explicit ticks improve readability of the logarithmic scale.
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1])
cbar.set_ticklabels([
    r"$10^{-4}$",
    r"$10^{-3}$",
    r"$10^{-2}$",
    r"$10^{-1}$",
])

plt.savefig(
    "kernel_reshaping_heatmap.pdf",
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.savefig(
    "kernel_reshaping_heatmap.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.show()
