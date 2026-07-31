import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================================================
# Raw data
# TTFT is in seconds
# TPOT is in milliseconds (converted to seconds for plotting)
# SLO is in percentage
# Lower is better for all metrics
# ============================================================

raw_data = {
    "Azure": {
        "TTFT": {
            "Throughput-Oriented": [0.18, 0.14, 0.34, 0.67],
            "TPOT-First":          [0.17, 0.13, 0.34, 0.56],
        },
        "TPOT": {
            "Throughput-Oriented": [29.0, 7.3, 69.5, 266.7],
            "TPOT-First":          [26.0, 7.1, 57.7, 233.6],
        },
        "SLO": {
            "Throughput-Oriented": [0.50, 6.83],   # [TTFT violation, TPOT violation]
            "TPOT-First":          [0.42, 5.38],
        },
    },

    "LongBench": {
        "TTFT": {
            "Throughput-Oriented": [17.38, 19.65, 30.23, 32.43],
            "TPOT-First":          [18.45, 18.80, 30.96, 32.21],
        },
        "TPOT": {
            "Throughput-Oriented": [104.7, 98.2, 203.5, 380.9],
            "TPOT-First":          [101.7, 96.1, 191.7, 366.2],
        },
        "SLO": {
            "Throughput-Oriented": [13.00, 10.40],
            "TPOT-First":          [13.57, 7.97],
        },
    },

    "BurstGPT": {
        "TTFT": {
            "Throughput-Oriented": [0.28, 0.12, 0.57, 1.52],
            "TPOT-First":          [0.30, 0.14, 0.80, 1.41],
        },
        "TPOT": {
            "Throughput-Oriented": [273.1, 5.9, 847.7, 3893.0],
            "TPOT-First":          [221.9, 5.9, 262.9, 2844.0],
        },
        "SLO": {
            "Throughput-Oriented": [4.33, 17.77],
            "TPOT-First":          [6.37, 9.58],
        },
    },
}

row_labels = ["Azure", "LongBench", "BurstGPT"]

metric_specs = {
    "TTFT": {
        "labels": ["Avg", "P50", "P90", "P99"],
        "title": "TTFT",
    },
    "TPOT": {
        "labels": ["Avg", "P50", "P90", "P99"],
        "title": "TPOT",
    },
    "SLO": {
        "labels": ["TTFT", "TPOT"],
        "title": "SLO Violations",
    },
}

# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 12,

    "axes.titlesize": 15.5,
    "axes.titleweight": "bold",

    "xtick.labelsize": 14.5,
    "ytick.labelsize": 13.5,

    "legend.fontsize": 17.0,

    "axes.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "figure.dpi": 180,
    "savefig.dpi": 400,
})

policy_colors = {
    "Throughput-Oriented": "#3B73B9",
    "TPOT-First": "#E07A33",
}

policy_markers = {
    "Throughput-Oriented": "o",
    "TPOT-First": "s",
}

connector_color = "#A9A9A9"

improve_color = "#1D7D3E"   # lower is better
degrade_color = "#B24633"
neutral_color = "#333333"

point_size = 30
connector_width = 1.45
horizontal_offset = 0.10

# ============================================================
# Helpers
# ============================================================

def convert_for_plot(metric, values):
    if metric == "TPOT":
        return [v / 1000.0 for v in values]   # ms -> s
    return list(values)

def pct_change(base, new):
    if base == 0:
        return None
    return 100.0 * (new - base) / base

def change_label(base, new):
    change = pct_change(base, new)
    if change is None:
        return ""
    if abs(change) < 0.5:
        return "≈0%"
    if change < 0:
        return f"{abs(change):.0f}%↓"
    return f"{change:.0f}%↑"

def change_color(base, new):
    change = pct_change(base, new)
    if change is None or abs(change) < 0.5:
        return neutral_color
    if change < 0:
        return improve_color
    return degrade_color

def sec_formatter(x, pos):
    if abs(x) < 1e-10:
        return "0"
    if abs(x) >= 1:
        if abs(x - round(x)) < 1e-8:
            return f"{int(round(x))}s"
        return f"{x:.1f}s"
    return f"{x:.1f}s"

def pct_formatter(x, pos):
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}%"
    return f"{x:.1f}%"

def apply_axis_format(ax, metric):
    if metric in ["TTFT", "TPOT"]:
        ax.yaxis.set_major_formatter(FuncFormatter(sec_formatter))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))

def set_limits(ax, metric, plotted_values):
    vals = np.asarray(plotted_values, dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = vmax - vmin
    if span == 0:
        span = max(abs(vmax), 1.0) * 0.1

    if metric == "SLO":
        lower = 0.0
        upper = vmax + max(0.22 * vmax, 0.8)
    else:
        lower = max(0.0, vmin - 0.16 * span)
        upper = vmax + 0.32 * span

    ax.set_ylim(lower, upper)

def draw_panel(ax, metric, base_raw, tpot_raw, xlabels):
    base_plot = convert_for_plot(metric, base_raw)
    tpot_plot = convert_for_plot(metric, tpot_raw)

    x = np.arange(len(xlabels), dtype=float)

    set_limits(ax, metric, base_plot + tpot_plot)
    apply_axis_format(ax, metric)

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    for i, (b_raw, t_raw, b_plot, t_plot) in enumerate(zip(base_raw, tpot_raw, base_plot, tpot_plot)):
        xb = x[i] - horizontal_offset
        xt = x[i] + horizontal_offset

        ax.plot(
            [xb, xt],
            [b_plot, t_plot],
            color=connector_color,
            linewidth=connector_width,
            solid_capstyle="round",
            zorder=1
        )

        ax.scatter(
            xb, b_plot,
            s=point_size,
            color=policy_colors["Throughput-Oriented"],
            marker=policy_markers["Throughput-Oriented"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3
        )

        ax.scatter(
            xt, t_plot,
            s=point_size,
            color=policy_colors["TPOT-First"],
            marker=policy_markers["TPOT-First"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3
        )

        label_y = max(b_plot, t_plot) + 0.050 * yspan
        ax.text(
            x[i], label_y,
            change_label(b_raw, t_raw),
            ha="center",
            va="bottom",
            fontsize=12.8,   # 更大的百分比变化数字
            fontweight="bold",
            color=change_color(b_raw, t_raw),
            bbox=dict(
                boxstyle="round,pad=0.05",
                facecolor="white",
                edgecolor="none",
                alpha=0.88
            ),
            zorder=5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.30)
    ax.tick_params(axis="x", length=2.5, width=0.9, pad=1.5)
    ax.tick_params(axis="y", length=2.5, width=0.9, pad=1.5)
    ax.margins(x=0.05)

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(7.35, 6.45),
)

metrics = ["TTFT", "TPOT", "SLO"]

for r, workload in enumerate(row_labels):
    for c, metric in enumerate(metrics):
        ax = axes[r, c]

        base_raw = raw_data[workload][metric]["Throughput-Oriented"]
        tpot_raw = raw_data[workload][metric]["TPOT-First"]

        draw_panel(
            ax=ax,
            metric=metric,
            base_raw=base_raw,
            tpot_raw=tpot_raw,
            xlabels=metric_specs[metric]["labels"],
        )

        if r == 0:
            ax.set_title(metric_specs[metric]["title"], pad=7)

        ax.set_ylabel("")

        if c == 0:
            ax.text(
                -0.26, 0.5, workload,
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=15.5,
                fontweight="bold",
            )

# ============================================================
# Legend
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        linestyle="none",
        marker=policy_markers["Throughput-Oriented"],
        markerfacecolor=policy_colors["Throughput-Oriented"],
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=8.2,   # 更大的图例标记
        label="Throughput-Oriented",
    ),
    Line2D(
        [0], [0],
        linestyle="none",
        marker=policy_markers["TPOT-First"],
        markerfacecolor=policy_colors["TPOT-First"],
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=8.2,   # 更大的图例标记
        label="TPOT-First",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.53, 1.02),
    ncol=2,
    frameon=False,
    handletextpad=0.50,
    columnspacing=1.8,
)

# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.12,
    right=0.992,
    bottom=0.085,
    top=0.88,
    wspace=0.24,   # 更大的列间距
    hspace=0.30,
)

# ============================================================
# Output
# ============================================================

plt.savefig("policy_comparison.pdf", bbox_inches="tight")
plt.show()