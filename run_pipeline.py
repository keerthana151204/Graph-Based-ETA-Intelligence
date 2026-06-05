"""End-to-end entrypoint: train + benchmark + plot.

Usage:
    python run_pipeline.py
Produces everything under outputs/: model_benchmark.csv, graph_advantage.json,
top5_bottleneck_hubs.csv, chronic_corridors.csv, node_centrality.csv,
eta_model.joblib, sage_embeddings.joblib, feature_meta.joblib, model_benchmark.png
"""
from src.train import run
from src.plot import make_plot

if __name__ == "__main__":
    run()
    make_plot()
