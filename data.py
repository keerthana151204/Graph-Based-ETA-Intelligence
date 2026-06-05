"""
Data loading, cleaning, and time/categorical feature engineering.

The only features produced here are *a-priori* — knowable at dispatch time.
Graph-derived features are added later (graph.py / graphsage.py) and are fit on
the training split only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and engineer a-priori features."""
    df = df.copy()

    # Valid OSRM and actual leg times only
    for col in [C.TARGET, "segment_osrm_time", "segment_osrm_distance", "osrm_time", "osrm_distance"]:
        df = df[df[col].notna()]
    df = df[(df["segment_osrm_time"] > 0) & (df[C.TARGET] > 0)]

    # Clip extreme upper tail of the target (data-entry artefacts), train-agnostic
    hi = df[C.TARGET].quantile(C.SETTINGS.target_clip_quantile)
    df = df[df[C.TARGET] <= hi]

    # Time features from the leg start time
    t = pd.to_datetime(df["od_start_time"], errors="coerce")
    df["hour"] = t.dt.hour.fillna(0).astype(int)
    df["day_of_week"] = t.dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)

    # Categorical / flag encodings
    df["route_type_enc"] = (df["route_type"].astype(str).str.upper() == "FTL").astype(int)
    df["is_cutoff"] = df["is_cutoff"].astype(int)
    df["cutoff_factor"] = pd.to_numeric(df["cutoff_factor"], errors="coerce").fillna(0)

    # Leg-level delay ratio — used ONLY to build the graph / SAGE target on train,
    # never as a model feature. Kept here for convenience, blocked in features.py.
    df["seg_delay_ratio"] = df[C.TARGET] / df["segment_osrm_time"]

    df = df.reset_index(drop=True)
    return df


def grouped_split(df: pd.DataFrame):
    """Leakage-safe split: all legs of a trip stay on the same side."""
    gss = GroupShuffleSplit(
        n_splits=1, test_size=C.SETTINGS.test_size, random_state=C.SETTINGS.random_state
    )
    train_idx, test_idx = next(gss.split(df, groups=df[C.GROUP_KEY]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
