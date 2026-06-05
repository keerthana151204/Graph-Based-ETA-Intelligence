"""Render the model-benchmark figure used in the README and memo."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from . import config as C

LABELS = {
    "0_osrm_as_is": "OSRM\n(today)",
    "1_baseline": "Baseline\nXGBoost",
    "2_centrality": "+ Centrality\n(graph)",
    "3_graphsage": "+ GraphSAGE\n(graph)",
}


def make_plot():
    df = pd.read_csv(C.OUTPUT_DIR / "model_benchmark.csv")
    df["label"] = df["model"].map(LABELS)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#b4b2a9", "#888780", "#1d9e75", "#0f6e56"]

    ax[0].bar(df["label"], df["MAE"], color=colors, edgecolor="white")
    ax[0].set_title("Mean Absolute Error (minutes) — lower is better")
    ax[0].set_ylabel("MAE (min)")
    for i, v in enumerate(df["MAE"]):
        ax[0].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)

    ax[1].bar(df["label"], df["within_15pct"], color=colors, edgecolor="white")
    ax[1].set_title("% of legs within 15% of actual — higher is better")
    ax[1].set_ylabel("within 15% (%)")
    for i, v in enumerate(df["within_15pct"]):
        ax[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("Graph-enhanced ETA vs leakage-safe baseline (grouped split by trip_uuid)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / "model_benchmark.png", dpi=130, bbox_inches="tight")
    print("wrote", C.OUTPUT_DIR / "model_benchmark.png")


if __name__ == "__main__":
    make_plot()
