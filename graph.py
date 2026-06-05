"""
Directed corridor graph + classical centrality features.

CRITICAL: the graph and every node statistic are built from the TRAINING legs
only, then mapped onto train and test. Test facilities unseen in training fall
back to the train median (or 0 for counts). This prevents the corridor/structure
statistics from carrying test-set information back into the features — the subtle
leakage path that a "build the graph on the whole dataframe" approach allows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

from . import config as C


def build_graph(train_df: pd.DataFrame) -> nx.DiGraph:
    """One node per facility, one edge per corridor; edge weight = median leg
    delay ratio on that corridor (the spec's 'median actual-vs-OSRM ratio')."""
    agg = (
        train_df.groupby([C.SOURCE, C.DEST])
        .agg(median_delay=("seg_delay_ratio", "median"),
             trips=("seg_delay_ratio", "size"))
        .reset_index()
    )
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r[C.SOURCE], r[C.DEST],
                   weight=float(r["median_delay"]), trips=int(r["trips"]))
    return G


def centrality_table(G: nx.DiGraph) -> pd.DataFrame:
    """Compute the centrality metrics named in the brief, per node."""
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    clustering = nx.clustering(G.to_undirected())

    nodes = list(G.nodes())
    return pd.DataFrame({
        "node": nodes,
        "betweenness": [betweenness.get(n, 0.0) for n in nodes],
        "pagerank": [pagerank.get(n, 0.0) for n in nodes],
        "in_degree": [in_deg.get(n, 0) for n in nodes],
        "out_degree": [out_deg.get(n, 0) for n in nodes],
        "clustering": [clustering.get(n, 0.0) for n in nodes],
    })


def attach_centrality(df: pd.DataFrame, cent: pd.DataFrame) -> pd.DataFrame:
    """Map node centrality onto each leg's source and destination, with
    train-median fallback for unseen nodes."""
    df = df.copy()
    cols = ["betweenness", "pagerank", "in_degree", "out_degree", "clustering"]
    lut = cent.set_index("node")[cols]
    fallback = lut.median(numeric_only=True)

    src = lut.reindex(df[C.SOURCE]).reset_index(drop=True)
    dst = lut.reindex(df[C.DEST]).reset_index(drop=True)
    for c in cols:
        df[f"src_{c}"] = src[c].fillna(fallback[c]).values
        df[f"dst_{c}"] = dst[c].fillna(fallback[c]).values
    return df


def top_bottleneck_hubs(train_df: pd.DataFrame, cent: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Rank hubs by a composite of structural centrality and SLA pain, with
    human-readable names — the table the strategy memo is built on."""
    name_lut = (
        train_df[[C.SOURCE, "source_name"]]
        .dropna().drop_duplicates(C.SOURCE)
        .set_index(C.SOURCE)["source_name"]
    )
    # SLA pain per source hub: breach = leg slower than OSRM (ratio > 1)
    pain = (
        train_df.assign(breach=(train_df["seg_delay_ratio"] > 1.0).astype(int))
        .groupby(C.SOURCE)
        .agg(sla_breach_rate=("breach", "mean"),
             median_delay=("seg_delay_ratio", "median"),
             trips=("seg_delay_ratio", "size"))
    )
    tbl = cent.set_index("node").join(pain, how="inner")
    tbl["hub_name"] = name_lut.reindex(tbl.index).fillna(tbl.index.to_series())
    # Composite score: structural reach x chronic pain x volume (log)
    tbl["bottleneck_score"] = (
        tbl["betweenness"].rank(pct=True)
        * tbl["sla_breach_rate"]
        * np.log1p(tbl["trips"])
    )
    out = tbl.sort_values("bottleneck_score", ascending=False).head(k).reset_index()
    return out[["node", "hub_name", "betweenness", "pagerank", "in_degree",
                "sla_breach_rate", "median_delay", "trips", "bottleneck_score"]]
