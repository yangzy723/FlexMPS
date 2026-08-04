from pathlib import Path

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

# Linear is zero for every system, so give it less horizontal space and
# redistribute that space to the informative operator columns.
column_widths = np.array([1.045, 1.045, 0.82, 1.045, 1.045])
column_edges = np.concatenate(([-0.5], -0.5 + np.cumsum(column_widths)))
column_centers = (column_edges[:-1] + column_edges[1:]) / 2

content_fontsize = 12.0

# Keep all source rows in the data arrays, but omit Salus visually.
display_rows = np.array([0, 1, 2, 4, 5, 6])
display_systems = [systems[index] for index in display_rows]

# Compress all-zero rows while preserving emphasis on rows with deviations.
row_heights = np.array([0.58, 0.58, 0.58, 0.58, 1.18, 1.18])
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

    "font.size": 9,
    "axes.titlesize": 12.5,
    "axes.labelsize": 9,
    "xtick.labelsize": content_fontsize,
    "ytick.labelsize": content_fontsize,

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
    display_data = data[display_rows]
    masked = np.ma.masked_equal(display_data, 0)

    image = ax.pcolormesh(
        column_edges,
        row_edges,
        masked,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.45,
    )

    ax.set_xlim(column_edges[0], column_edges[-1])
    ax.set_ylim(row_edges[-1], 0)

    ax.set_title(title, pad=4, fontweight="normal")

    ax.set_xticks(column_centers)
    ax.set_xticklabels(operators)
    ax.tick_params(axis="x", pad=1.0)
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_rotation(25)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("top")
        label.set_fontsize(content_fontsize)
        shift_points = -1.5 if index == 0 else 6.0
        label.set_transform(
            label.get_transform()
            + mpl.transforms.ScaledTranslation(
                shift_points / 72,
                0,
                ax.figure.dpi_scale_trans,
            )
        )

    ax.set_yticks(row_centers)
    if show_ylabels:
        ax.set_yticklabels(display_systems)
        yticklabels = ax.get_yticklabels()
        ax.tick_params(axis="y", pad=3.5)
        for label in yticklabels:
            label.set_rotation(15)
            label.set_rotation_mode("anchor")
            label.set_ha("right")
            label.set_va("center")
            label.set_fontsize(content_fontsize)
            label.set_fontweight("normal")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)

    # Separate fixed-structure systems from reshaping systems.
    ax.axhline(
        row_edges[4],
        color="#333333",
        linewidth=0.8,
        linestyle=(0, (2.5, 1.5)),
    )

    # Cell annotations.
    for row in range(display_data.shape[0]):
        for col in range(display_data.shape[1]):
            value = display_data[row, col]

            ax.text(
                column_centers[col],
                row_centers[row],
                format_value(value),
                ha="center",
                va="center",
                fontsize=content_fontsize,
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
fig = plt.figure(figsize=(7.05, 2.18))
grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[1, 1],
    left=0.155,
    right=0.990,
    bottom=0.305,
    top=0.865,
    wspace=0.075,
)
axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
]
axes[1].sharey(axes[0])

draw_panel(
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

output_dir = Path(__file__).resolve().parent

plt.savefig(
    output_dir / "kernel_reshaping.pdf",
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.show()
