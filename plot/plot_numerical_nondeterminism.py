import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


PRECISION_STYLE = {
    "FP16": {
        "color": "#1D4ED8",   # deep blue
        "marker": "o",
    },
    "BF16": {
        "color": "#B71C1C",   # deep red
        "marker": "s",
    },
}


def _load_deviation(
    precision_results: dict[str, Any],
    keys: tuple[str, ...],
    precision: str,
) -> np.ndarray:
    """Load a deviation metric while supporting legacy JSON key names."""
    for key in keys:
        if key in precision_results:
            values = np.asarray(precision_results[key], dtype=float)
            if values.ndim != 1:
                raise ValueError(
                    f"{precision}.{key} must be a one-dimensional array."
                )
            return values

    raise KeyError(
        f"Missing one of {keys} for precision {precision}."
    )


def _validate_deviations(
    mean_values: np.ndarray,
    max_values: np.ndarray,
    num_configs: int,
    precision: str,
) -> None:
    """Validate numerical-deviation data before plotting."""
    if len(mean_values) != num_configs:
        raise ValueError(
            f"{precision}: expected {num_configs} mean values, "
            f"received {len(mean_values)}."
        )

    if len(max_values) != num_configs:
        raise ValueError(
            f"{precision}: expected {num_configs} maximum values, "
            f"received {len(max_values)}."
        )

    if not np.all(np.isfinite(mean_values)):
        raise ValueError(f"{precision}: mean deviations contain invalid values.")

    if not np.all(np.isfinite(max_values)):
        raise ValueError(f"{precision}: maximum deviations contain invalid values.")

    if np.any(mean_values <= 0) or np.any(max_values <= 0):
        raise ValueError(
            "All plotted deviations must be positive because the y-axis is "
            "logarithmic. Omit the zero-deviation reference configuration C "
            "rather than replacing zero with an artificial epsilon."
        )

    if np.any(max_values < mean_values):
        raise ValueError(
            f"{precision}: maximum deviation cannot be smaller than "
            "mean deviation."
        )


def plot_reshaping_deviation(
    block_configs: list[int],
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot numerical deviations introduced by reshaped execution structures.

    Filled markers show mean absolute elementwise deviation.
    Open markers show maximum absolute elementwise deviation.
    Vertical stems connect the two statistics for each configuration.
    """
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": [
            "STIXGeneral",
            "Times New Roman",
            "DejaVu Serif",
            "Liberation Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 7.1,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.7,

        # Axes
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

        # Export
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not block_configs:
        raise ValueError("block_configs must not be empty.")

    config_labels = [str(config) for config in block_configs]
    x = np.arange(len(block_configs), dtype=float)

    precision_order = [
        precision
        for precision in ("FP16", "BF16")
        if precision in results
    ]

    if not precision_order:
        raise ValueError("No FP16 or BF16 results were found.")

    # Slight horizontal separation without implying continuity.
    if len(precision_order) == 1:
        offsets = {precision_order[0]: 0.0}
    else:
        offsets = {
            "FP16": -0.10,
            "BF16": 0.10,
        }

    # Narrower figure for the right panel of a two-panel paper figure.
    fig, ax = plt.subplots(
        figsize=(2, 1.88),
        constrained_layout=True,
    )

    all_positive_values: list[float] = []

    for precision in precision_order:
        precision_results = results[precision]

        mean_values = _load_deviation(
            precision_results,
            ("mean_ad", "mad"),
            precision,
        )
        max_values = _load_deviation(
            precision_results,
            ("max_ad",),
            precision,
        )

        _validate_deviations(
            mean_values,
            max_values,
            len(block_configs),
            precision,
        )

        all_positive_values.extend(mean_values.tolist())
        all_positive_values.extend(max_values.tolist())

        style = PRECISION_STYLE[precision]
        positions = x + offsets.get(precision, 0.0)

        # Mean-to-maximum range.
        ax.vlines(
            positions,
            mean_values,
            max_values,
            color=style["color"],
            linewidth=1.0,
            alpha=0.72,
            zorder=2,
        )

        # Mean: filled marker.
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

        # Maximum: open marker.
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
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)

    ax.set_xlabel(r"Grid partitions in $C_t$", labelpad=0.8)
    ax.set_ylabel(r"Absolute deviation from $F_K(X,C)$", labelpad=0.8)

    ax.tick_params(axis="x", which="both", direction="in", pad=1.2)
    ax.tick_params(
        axis="y", which="both", direction="in", pad=1.2, labelrotation=0
    )

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    minimum = min(all_positive_values)
    maximum = max(all_positive_values)
    log_min = np.log10(minimum)
    log_max = np.log10(maximum)
    padding = max(0.08 * (log_max - log_min), 0.07)

    ax.set_ylim(
        10 ** (log_min - padding),
        10 ** (log_max + padding),
    )

    ax.set_xlim(-0.42, len(block_configs) - 0.58)

    # Only major horizontal guides.
    ax.grid(
        axis="y",
        which="major",
        color="#C6CBD1",
        linestyle=(0, (3, 2)),
        linewidth=0.58,
        alpha=0.68,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="minor", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # Compact legend row.
    precision_handles = [
        Line2D(
            [0],
            [0],
            marker=PRECISION_STYLE[precision]["marker"],
            linestyle="none",
            markerfacecolor=PRECISION_STYLE[precision]["color"],
            markeredgecolor=PRECISION_STYLE[precision]["color"],
            markeredgewidth=0.8,
            markersize=3.8,
            label=precision,
        )
        for precision in precision_order
    ]

    statistic_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#4A4A4A",
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.75,
            markersize=3.7,
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.95,
            markersize=3.7,
            label="Max",
        ),
    ]

    ax.legend(
        handles=precision_handles + statistic_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.68,
        handletextpad=0.24,
        columnspacing=0.52,
    )

    pdf_path = output_dir / "reshaping_numerical_deviation.pdf"
    png_path = output_dir / "reshaping_numerical_deviation.png"

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.012,
    )
    fig.savefig(
        png_path,
        format="png",
        bbox_inches="tight",
        pad_inches=0.012,
        dpi=300,
    )
    plt.close(fig)

    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")


def main() -> None:
    plot_dir = Path(__file__).resolve().parent
    result_path = (
        plot_dir
        / "determinism"
        / "result_determinism.txt"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Result file does not exist: {result_path}"
        )

    with result_path.open("r", encoding="utf-8") as result_file:
        data = json.load(result_file)

    if "configs" not in data or "results" not in data:
        raise KeyError(
            "The result file must contain 'configs' and 'results'."
        )

    plot_reshaping_deviation(
        block_configs=data["configs"],
        results=data["results"],
        output_dir=plot_dir,
    )


if __name__ == "__main__":
    main()