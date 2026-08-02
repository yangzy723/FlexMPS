#!/usr/bin/env python3
"""Plot GreenCtx lifecycle/pooling/policy scenario latency from CSV."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT = HERE / "greenctx_scheduling_overhead.csv"
OUTPUT = HERE / "greenctx_scheduling_overhead.pdf"
SCENARIOS = [
    ("native_gemm_baseline", "Native"),
    ("greenctx_no_pool_policy_350", "GreenCtx w/o Pool"),
    ("greenctx_pool_policy_350", "GreenCtx + Flat Pool"),
    ("greenctx_pool_policy_32", "GreenCtx + Hierarchy Pool"),
]


def load_results():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        by_scenario = {row["scenario"]: row for row in csv.DictReader(handle)}
    missing = [name for name, _ in SCENARIOS if name not in by_scenario]
    if missing:
        raise RuntimeError(f"missing scenarios in {INPUT.name}: {missing}")
    labels = [label for _, label in SCENARIOS]
    latency = [float(by_scenario[name]["wall_time_p95_ms"])
               for name, _ in SCENARIOS]
    overhead = [float(by_scenario[name]["overhead_vs_baseline_pct"])
                for name, _ in SCENARIOS]
    return labels, latency, overhead


def style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="in", length=3.5, width=0.9)
    axis.grid(axis="y", linestyle="--", linewidth=0.7,
              color="#b8b8b8", alpha=0.55)
    axis.set_axisbelow(True)


def main():
    labels, latency_ms, overhead_pct = load_results()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.2,
        "axes.labelweight": "normal",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axis = plt.subplots(figsize=(3.35, 2.585))
    x_positions = [index * 0.72 for index in range(len(labels))]
    hatches = ["", "////", "\\\\", "...."]
    colors = ["#4C78A8", "#E45756", "#F2CF5B", "#59A14F"]
    bars = []
    for x_pos, value, label, hatch, color in zip(
            x_positions, latency_ms, labels, hatches, colors):
        bar = axis.bar(x_pos, value, width=0.42, label=label, color=color,
                       edgecolor="black", linewidth=0.8, hatch=hatch)
        bars.append(bar[0])

    axis.set_ylabel("p95 latency (ms)", fontsize=9.5)
    axis.set_xticks([])
    axis.set_xlim(x_positions[0] - 0.34, x_positions[-1] + 0.34)
    axis.set_ylim(0, 2.62)
    style_axes(axis)

    for index, (bar, value, overhead) in enumerate(
            zip(bars, latency_ms, overhead_pct)):
        text = f"{value:.3f}" if index == 0 else f"{value:.3f}\n(+{overhead:.1f}%)"
        axis.annotate(text,
                      xy=(bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 5), textcoords="offset points",
                      ha="center", va="bottom", fontsize=9.5,
                      fontweight="semibold", linespacing=1.0)

    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.52), ncol=2,
                frameon=False, handlelength=1.35, columnspacing=0.75,
                handletextpad=0.35)
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.13, top=0.63)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


if __name__ == "__main__":
    main()
