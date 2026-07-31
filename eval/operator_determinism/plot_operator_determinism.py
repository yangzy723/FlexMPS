from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PANELS = (
    (r"(a) Reduction Kernel ($\times 10^{-2}$)", ROOT / "reduction.csv", 64, 1e2, 16),
    (r"(b) Split-K GEMM Kernel ($\times 10^{-3}$)", ROOT / "splitk_gemm.csv", 1, 1e3, 15),
)
OUTPUT_PNG = ROOT / "operator_determinism_distribution.png"
OUTPUT_PDF = ROOT / "operator_determinism_distribution.pdf"

RESHAPED_COLOR = "#D1A45F"
RESHAPED_EDGE = "#8A642E"
PROTEUS_COLOR = "#C73737"
PROTEUS_DASH = (0, (7.0, 2.2, 1.4, 2.2))

REQUIRED_COLUMNS = {
    "dtype",
    "test_shape",
    "max_abs",
    "accumulation",
    "reference_nan_count",
    "test_nan_count",
    "reference_inf_count",
    "test_inf_count",
}
NONFINITE_COLUMNS = [
    "reference_nan_count",
    "test_nan_count",
    "reference_inf_count",
    "test_inf_count",
]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 11,
        "legend.fontsize": 19,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#303030",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 400,
    })


def load_reshaped_drift(path: Path, baseline: int, scale: float) -> np.ndarray:
    dataframe = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    accumulation = dataframe["accumulation"].astype(str).str.strip().str.lower()
    if set(accumulation) != {"fp32"}:
        raise ValueError(f"{path.name} was not produced with FP32 accumulation")
    if dataframe[NONFINITE_COLUMNS].to_numpy().sum() != 0:
        raise ValueError(f"{path.name} contains non-finite outputs")

    selected = dataframe[
        (dataframe["dtype"].str.strip().str.lower() == "fp32")
        & (dataframe["test_shape"].astype(int) != baseline)
    ]
    if selected.empty:
        raise ValueError(f"{path.name} contains no reshaped FP32 trials")

    return selected["max_abs"].to_numpy(dtype=float) * scale


def draw_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    number_of_bins: int,
) -> None:
    maximum = float(values.max())
    edges = np.linspace(0.0, maximum * 1.025, number_of_bins + 1)
    weights = np.full(values.size, 100.0 / values.size)

    ax.hist(
        values,
        bins=edges,
        weights=weights,
        color=RESHAPED_COLOR,
        edgecolor=RESHAPED_EDGE,
        alpha=0.84,
        linewidth=1.05,
        rwidth=0.88,
        zorder=3,
    )
    ax.axvline(
        0.0,
        color=PROTEUS_COLOR,
        linestyle=PROTEUS_DASH,
        linewidth=2.6,
        dash_capstyle="round",
        zorder=5,
    )

    ax.set_title(title, pad=12)
    # Separate the zero-drift reference from the left spine.
    ax.set_xlim(-maximum * 0.045, maximum * 1.055)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.30, zorder=0)
    ax.tick_params(axis="x", length=4.0, width=1.0, pad=4)
    ax.tick_params(axis="y", length=3.5, width=0.9)
    ax.set_axisbelow(True)


def build_legend_handles():
    return [
        Patch(
            facecolor=RESHAPED_COLOR,
            edgecolor=RESHAPED_EDGE,
            alpha=0.84,
            label="Kernel Reshaping",
        ),
        Line2D(
            [0],
            [0],
            color=PROTEUS_COLOR,
            linestyle=PROTEUS_DASH,
            linewidth=2.6,
            dash_capstyle="round",
            label="PROTEUS",
        ),
    ]


def main() -> None:
    configure_style()
    values_by_panel = [
        load_reshaped_drift(path, baseline, scale)
        for _, path, baseline, scale, _ in PANELS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35), sharey=True)
    for ax, values, (title, _, _, _, bins) in zip(
        axes,
        values_by_panel,
        PANELS,
    ):
        draw_histogram(ax, values, title, bins)

    axes[0].set_ylabel("Trial Frequency (%)", fontsize=19, labelpad=8)
    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        handletextpad=0.65,
        columnspacing=1.8,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.99,
        bottom=0.15,
        top=0.67,
        wspace=0.08,
    )

    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved operator determinism plots to {OUTPUT_PNG} and {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
