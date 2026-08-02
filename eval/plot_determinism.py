from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import torch


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "determinism" / "determinism_data.pt"
OUTPUT_PATH = ROOT / "determinism.pdf"

RESHAPING_COLOR = "#1f77b4"
PROTEUS_COLOR = "#d62728"
FLIP_COLOR = "#ff7f0e"
ZERO_COLOR = "gray"

LINE_WIDTH = 2.0
MARKER_SIZE = 10
FLIP_MARKER_SIZE = 180


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 32,
        "axes.labelsize": 36,
        "axes.labelweight": "bold",
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })


def draw_stage(
    ax: plt.Axes,
    reshaping_drift,
    proteus_drift,
    flip_indices,
    title: str,
) -> None:
    trials = range(len(reshaping_drift))

    ax.axhline(
        0,
        color=ZERO_COLOR,
        linewidth=4.0,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    ax.plot(
        trials,
        proteus_drift,
        color=PROTEUS_COLOR,
        alpha=0.9,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    ax.plot(
        trials,
        reshaping_drift,
        color=RESHAPING_COLOR,
        alpha=0.85,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        zorder=3,
    )

    if flip_indices:
        flip_values = [reshaping_drift[index] for index in flip_indices]
        ax.scatter(
            flip_indices,
            flip_values,
            marker="X",
            s=FLIP_MARKER_SIZE,
            color=FLIP_COLOR,
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )
        for index in flip_indices:
            ax.axvspan(
                index - 0.5,
                index + 0.5,
                color=FLIP_COLOR,
                alpha=0.15,
                zorder=0,
            )

    ax.set_title(title, fontsize=40, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def build_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=RESHAPING_COLOR,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=18,
            label="Kernel Reshaping",
        ),
        Line2D(
            [0],
            [0],
            color=PROTEUS_COLOR,
            linewidth=LINE_WIDTH,
            marker="s",
            markersize=18,
            label="proteus",
        ),
        Line2D(
            [0],
            [0],
            color=ZERO_COLOR,
            linewidth=4.0,
            linestyle="--",
            alpha=0.7,
            label="Bitwise Equality",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            marker="X",
            markerfacecolor=FLIP_COLOR,
            markeredgecolor="black",
            markersize=22,
            markeredgewidth=1.5,
            label="Argmax Flip Occurred",
        ),
    ]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Determinism data not found: {DATA_PATH}")

    data = torch.load(DATA_PATH, map_location="cpu")
    flip_indices = [int(index) for index in data.get("flip_both_indices", [])]

    configure_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 11.2), sharex=True)

    draw_stage(
        ax1,
        data["drift_both_logits"],
        data["drift_bs_only_logits"],
        flip_indices,
        "Stage 1: LM Head MatMul Logit Drift",
    )
    draw_stage(
        ax2,
        data["drift_both_probs"],
        data["drift_bs_only_probs"],
        flip_indices,
        "Stage 2: Softmax Probability Drift",
    )

    fig.supylabel(
        "Max Absolute Drift",
        fontweight="bold",
        fontsize=36,
        x=0.018,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.997,
        bottom=0.075,
        top=0.825,
        hspace=0.18,
    )
    fig.legend(
        handles=build_legend_handles(),
        loc="upper left",
        ncol=4,
        bbox_to_anchor=(0.085, 0.935, 0.912, 0.06),
        mode="expand",
        frameon=False,
        fontsize=38,
        handlelength=1.35,
        handletextpad=0.2,
        borderpad=0.0,
        borderaxespad=0.0,
        columnspacing=0.7,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved determinism plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
