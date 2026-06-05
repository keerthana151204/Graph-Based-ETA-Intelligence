"""
Central configuration for the Delhivery Graph-ETA pipeline.

Every modelling decision that affects leakage or reproducibility lives here so it
is auditable in one place, not scattered across notebooks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "delivery_data.csv"
OUTPUT_DIR = ROOT / "outputs"

# --- Grain & grouping ------------------------------------------------------
# Each row is one leg (segment) of a trip. trip_uuid groups legs that belong to
# the same shipment. We MUST split on this key, never on the row, otherwise legs
# of the same trip leak across train/test and inflate the score.
GROUP_KEY = "trip_uuid"
SOURCE = "source_center"
DEST = "destination_center"

# --- Target ---------------------------------------------------------------
# Predict the actual time of a leg. OSRM's prediction for the same leg is given
# as a feature, so the model learns the *correction* OSRM systematically misses.
TARGET = "segment_actual_time"
OSRM_REFERENCE = "segment_osrm_time"  # naive baseline = use this as the prediction

# --- Leakage blocklist ----------------------------------------------------
# These are derived from the actual outcome (or are post-hoc) and must NEVER
# enter the feature matrix. This single list is the project's anti-leakage spine.
LEAKAGE_COLS = [
    "segment_actual_time",        # the target itself
    "segment_factor",             # = segment_actual_time / segment_osrm_time
    "actual_time",                # trip-level actual outcome
    "factor",                     # = actual_time / osrm_time
    "actual_distance_to_destination",  # measured post-hoc -> excluded to be safe
    "start_scan_to_end_scan",     # ~= actual elapsed time
    "od_end_time",                # known only on completion
    "cutoff_timestamp",
]

# A-priori features (known at dispatch time)
BASELINE_FEATURES = [
    "segment_osrm_time",
    "segment_osrm_distance",
    "osrm_time",
    "osrm_distance",
    "route_type_enc",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak",
    "is_cutoff",
    "cutoff_factor",
]

CENTRALITY_FEATURES = [
    "src_betweenness", "src_pagerank", "src_in_degree", "src_out_degree", "src_clustering",
    "dst_betweenness", "dst_pagerank", "dst_in_degree", "dst_out_degree", "dst_clustering",
]


@dataclass
class Settings:
    test_size: float = 0.2
    random_state: int = 42
    target_clip_quantile: float = 0.995  # drop only the extreme upper tail
    # GraphSAGE
    sage_embed_dim: int = 16
    sage_hidden_dim: int = 32
    sage_epochs: int = 120
    sage_lr: float = 0.02
    sage_seed: int = 7
    # XGBoost
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=400, max_depth=7, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=2.0, n_jobs=4, random_state=42, tree_method="hist",
    ))


SETTINGS = Settings()
