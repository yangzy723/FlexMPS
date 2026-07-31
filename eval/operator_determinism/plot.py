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

FP32_COLOR = "#3F78A8"
FP32_EDGE = "#28567F"
PROTEUS_COLOR = "#303030"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 400,
})


def load_fp32(path: Path, baseline: int, scale: float) -> np.ndarray:
    dataframe = pd.read_csv(path)
    required = {
        "dtype", "test_shape", "max_abs", "accumulation",
        "reference_nan_count", "test_nan_count",
        "reference_inf_count", "test_inf_count",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if set(dataframe["accumulation"]) != {"fp32"}:
        raise ValueError(f"{path.name} was not produced with FP32 accumulation")
    if dataframe[[
        "reference_nan_count", "test_nan_count",
        "reference_inf_count", "test_inf_count",
    ]].to_numpy().sum() != 0:
        raise ValueError(f"{path.name} contains non-finite outputs")

    selected = dataframe[
        (dataframe["dtype"].str.strip().str.lower() == "fp32") &
        (dataframe["test_shape"].astype(int) != baseline)
    ]
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
        color=FP32_COLOR,
        edgecolor=FP32_EDGE,
        alpha=0.78,
        linewidth=1.15,
        rwidth=0.90,
        zorder=3,
    )
    ax.axvline(
        0.0,
        color=PROTEUS_COLOR,
        linestyle="--",
        linewidth=2.2,
        zorder=5,
    )

    ax.set_title(title, pad=8)
    ax.set_xlim(-maximum * 0.025, maximum * 1.055)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.30, zorder=0)
    ax.tick_params(axis="x", length=4.0, width=1.0, pad=4)
    ax.tick_params(axis="y", length=3.5, width=0.9)
    ax.set_axisbelow(True)


values_by_panel = [
    load_fp32(path, baseline, scale)
    for _, path, baseline, scale, _ in PANELS
]

fig, axes = plt.subplots(1, 2, figsize=(8.15, 2.8), sharey=True)
for ax, values, (title, _, _, _, bins) in zip(axes, values_by_panel, PANELS):
    draw_histogram(ax, values, title, bins)

axes[0].set_ylabel("Trials per bin (%)", fontsize=12, labelpad=5)
fig.legend(
    handles=[
        Patch(
            facecolor=FP32_COLOR,
            edgecolor=FP32_EDGE,
            alpha=0.78,
            label="FP32 / Reshaped structure",
        ),
        Line2D(
            [0], [0],
            color=PROTEUS_COLOR,
            linestyle="--",
            linewidth=2.2,
            label="PROTEUS / Fixed structure: zero drift",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    frameon=False,
    handlelength=2.4,
    handletextpad=0.6,
    columnspacing=1.5,
)

fig.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.69, wspace=0.15)
fig.savefig(OUTPUT_PNG, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close(fig)
