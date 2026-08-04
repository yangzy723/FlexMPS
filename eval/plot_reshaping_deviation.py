import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


PRECISION_STYLE = {
    "FP16": {"color": "#1D4ED8", "marker": "o"},
    "BF16": {"color": "#B71C1C", "marker": "s"},
}


def _deviations(results, precision, num_configs):
    """Load and validate mean and maximum deviations for one precision."""
    precision_results = results[precision]
    mean_values = np.asarray(precision_results["mad"], dtype=float)
    max_values = np.asarray(precision_results["max_ad"], dtype=float)

    for name, values in (("mad", mean_values), ("max_ad", max_values)):
        if values.shape != (num_configs,):
            raise ValueError(
                f"{precision}.{name} must contain {num_configs} values."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{precision}.{name} contains invalid values.")
        if np.any(values <= 0):
            raise ValueError(
                f"{precision}.{name} must be positive for a logarithmic axis."
            )

    if np.any(max_values < mean_values):
        raise ValueError(
            f"{precision}.max_ad cannot be smaller than {precision}.mad."
        )

    return mean_values, max_values


def plot_reshaping_deviation(block_configs, results, output_dir):
    """Plot mean and maximum numerical deviations for each grid partition."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "STIXGeneral",
            "Times New Roman",
            "DejaVu Serif",
            "Liberation Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 7.1,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#202020",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.72,
        "ytick.major.width": 0.72,
        "xtick.minor.width": 0.52,
        "ytick.minor.width": 0.52,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    if not block_configs:
        raise ValueError("block_configs must not be empty.")

    precisions = tuple(
        precision for precision in PRECISION_STYLE if precision in results
    )
    if not precisions:
        raise ValueError("No FP16 or BF16 results were found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(block_configs), dtype=float)
    offsets = np.linspace(-0.10, 0.10, len(precisions)) if len(precisions) > 1 else [0]
    fig, ax = plt.subplots(figsize=(2.00, 1.70), constrained_layout=True)
    plotted_values = []

    for precision, offset in zip(precisions, offsets):
        mean_values, max_values = _deviations(
            results, precision, len(block_configs)
        )
        plotted_values.extend((mean_values, max_values))

        style = PRECISION_STYLE[precision]
        positions = x + offset
        ax.vlines(
            positions,
            mean_values,
            max_values,
            color=style["color"],
            linewidth=1.0,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            positions,
            mean_values,
            marker=style["marker"],
            s=23,
            facecolor=style["color"],
            edgecolor=style["color"],
            linewidth=0.72,
            zorder=4,
        )
        ax.scatter(
            positions,
            max_values,
            marker=style["marker"],
            s=26,
            facecolor="white",
            edgecolor=style["color"],
            linewidth=1.05,
            zorder=5,
        )

    ax.set_yscale("log")
    ax.set_xticks(x, [str(config) for config in block_configs])
    ax.set_xlabel(
        r"Grid Partitions in $\widetilde{C}_k$",
        fontsize=10.2,
        labelpad=0.8,
    )
    ax.set_ylabel("Absolute Deviation from Ref.", labelpad=0.8)
    ax.tick_params(axis="x", which="both", pad=1.2, labelsize=9.5)
    ax.tick_params(
        axis="y",
        which="both",
        pad=1.2,
        labelrotation=0,
        labelsize=9.5,
    )

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    values = np.concatenate(plotted_values)
    log_min, log_max = np.log10([values.min(), values.max()])
    padding = max(0.08 * (log_max - log_min), 0.07)
    ax.set_ylim(10 ** (log_min - padding), 10 ** (log_max + padding))
    ax.set_xlim(-0.28, len(block_configs) - 0.72)

    ax.grid(
        axis="y",
        which="major",
        color="#C6CBD1",
        linestyle=(0, (3, 2)),
        linewidth=0.58,
        alpha=0.68,
        zorder=0,
    )
    ax.spines[["top", "right"]].set_visible(False)

    precision_handles = [
        Line2D(
            [0],
            [0],
            marker=PRECISION_STYLE[precision]["marker"],
            linestyle="none",
            markerfacecolor=PRECISION_STYLE[precision]["color"],
            markeredgecolor=PRECISION_STYLE[precision]["color"],
            markeredgewidth=0.8,
            markersize=4.6,
            label=precision,
        )
        for precision in precisions
    ]
    statistic_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="#4A4A4A", markeredgecolor="#4A4A4A",
            markeredgewidth=0.75, markersize=4.5, label="Mean",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="white", markeredgecolor="#4A4A4A",
            markeredgewidth=0.95, markersize=4.5, label="Max",
        ),
    ]
    ax.legend(
        handles=precision_handles + statistic_handles,
        loc="lower left",
        bbox_to_anchor=(-0.12, 1.01),
        ncol=4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.60,
        handletextpad=0.20,
        columnspacing=0.42,
    )

    pdf_path = output_dir / "reshaping_deviation.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)

    print(f"Generated: {pdf_path}")


def main():
    plot_dir = Path(__file__).resolve().parent
    result_path = plot_dir / "determinism" / "result_determinism.txt"

    if not result_path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {result_path}")

    with result_path.open("r", encoding="utf-8") as result_file:
        data = json.load(result_file)

    try:
        block_configs = data["configs"]
        results = data["results"]
    except KeyError as error:
        raise KeyError("The result file must contain configs and results.") from error

    plot_reshaping_deviation(block_configs, results, plot_dir)


if __name__ == "__main__":
    main()
