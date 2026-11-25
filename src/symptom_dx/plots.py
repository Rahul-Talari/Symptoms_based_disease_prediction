"""Bar charts of the evaluation results in results/metrics.csv and results/inference_time.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMPLEXITIES = ["Simple", "Moderate", "Complex"]


def draw_bars(ax, df: pd.DataFrame, title: str, ylabel: str) -> None:
    """Grouped bars: one group per technique, one bar per symptom complexity."""
    table = df.pivot(index="technique", columns="complexity", values="value")
    table = table[[c for c in COMPLEXITIES if c in table.columns]]
    table.plot.bar(ax=ax, rot=0, width=0.8, legend=False)
    ax.set_title(title)
    ax.set_xlabel("Technique")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def plot_metrics(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    metrics = df["metric"].unique()
    rows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.5 * rows))
    for ax, metric in zip(axes.flat, metrics):
        draw_bars(ax, df[df["metric"] == metric], metric, metric)
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    add_legend(fig, axes.flat[0])
    save(fig, out_path)


def plot_inference_time(csv_path: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw_bars(ax, pd.read_csv(csv_path), "Inference time", "Seconds")
    add_legend(fig, ax)
    save(fig, out_path)


def add_legend(fig, ax) -> None:
    """One shared legend below the charts, so it never covers a bar."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Symptom complexity",
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        frameon=False,
    )


def save(fig, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")  # "tight" keeps the legend below the charts
    plt.close(fig)
