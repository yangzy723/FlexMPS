#!/usr/bin/env python3
"""Plot separately measured GreenCtx and MPS context-pool residency."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "pool_memory_overhead.pdf"


def load_levels(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["statistic"] == "resident_after_level"
        ]
    rows.sort(key=lambda row: int(row["level"]))
    return {
        "level": [int(row["level"]) for row in rows],
        "gpu": [float(row["gpu_increment_from_empty_mib"]) for row in rows],
        "rss": [float(row["rss_increment_from_empty_mib"]) for row in rows],
    }


def style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="in", length=3.5, width=0.9)
    axis.grid(axis="y", linestyle="--", linewidth=0.7,
              color="#b8b8b8", alpha=0.55)
    axis.set_axisbelow(True)


def main():
    green = load_levels(HERE / "greenctx_memory_overhead.csv")
    mps = load_levels(HERE / "mps_memory_overhead.csv")
    if green["level"] != mps["level"]:
        raise RuntimeError("GreenCtx and MPS CSV files have different hierarchy levels")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axis = plt.subplots(figsize=(3.35, 2.125))
    green_color = "#E45756"
    mps_color = "#4C78A8"

    axis.plot(green["level"], green["gpu"], marker="o", markersize=4.6,
              linewidth=1.8, color=green_color, label="GreenCtx GPU")
    axis.plot(mps["level"], mps["gpu"], marker="s", markersize=4.4,
              linewidth=1.8, color=mps_color, label="MPS-Pool GPU")
    axis.plot(green["level"], green["rss"], marker="o", markersize=3.8,
              linewidth=1.45, linestyle="--", color=green_color,
              label="GreenCtx RSS")
    axis.plot(mps["level"], mps["rss"], marker="s", markersize=3.6,
              linewidth=1.45, linestyle="--", color=mps_color,
              label="MPS-Pool RSS")

    axis.set_xlabel("Hierarchy level")
    axis.set_ylabel("Memory overhead (MiB)")
    axis.set_xticks(green["level"])
    axis.set_xlim(0.8, max(green["level"]) + 1.25)
    axis.set_ylim(0, 2700)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(3, 3),
                          useMathText=True)
    axis.yaxis.get_offset_text().set_visible(False)
    style_axes(axis)
    axis.text(0.015, 0.965, r"$\times 10^3$", transform=axis.transAxes,
              ha="left", va="top", fontsize=8.5, fontweight="semibold")

    handles, labels = axis.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    axis.legend([handles[index] for index in order],
                [labels[index] for index in order],
                loc="upper center", bbox_to_anchor=(0.57, 0.985),
                bbox_transform=figure.transFigure, ncol=2, frameon=False,
                handlelength=1.65, columnspacing=0.75, handletextpad=0.35)

    for values, offset in ((green["gpu"], 0), (mps["gpu"], 0),
                           (green["rss"], -4), (mps["rss"], 0)):
        axis.annotate(f"{values[-1]:.0f}",
                      xy=(green["level"][-1], values[-1]),
                      xytext=(8, offset), textcoords="offset points",
                      ha="left", va="center", fontsize=8.5,
                      fontweight="semibold")

    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.72)
    figure.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


if __name__ == "__main__":
    main()
