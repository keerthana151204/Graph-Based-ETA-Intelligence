"""
End-to-end training & benchmark.

Runs four predictors on the SAME leakage-safe grouped split and SAME target:
  0. OSRM as-is        (use segment_osrm_time as the prediction — the system today)
  1. Baseline XGBoost  (a-priori trip/OSRM/time features only)
  2. + Centrality      (Surbhi's approach: add classical graph centrality)
  3. + GraphSAGE       (add from-scratch GraphSAGE node embeddings)

The "graph advantage" is therefore measured against a like-for-like baseline,
not asserted.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import config as C
from . import data as D
from . import graph as Gm
from . import graphsage as S
from .evaluate import score


def _fit_xgb(Xtr, ytr):
    m = XGBRegressor(**C.SETTINGS.xgb_params)
    m.fit(Xtr, ytr)
    return m


def run(verbose=True):
    out = C.OUTPUT_DIR
    out.mkdir(exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    log("1/6  Loading + cleaning ...")
    df = D.clean(D.load_raw())
    log(f"     legs={len(df):,}  trips={df[C.GROUP_KEY].nunique():,}  "
        f"facilities={pd.concat([df[C.SOURCE], df[C.DEST]]).nunique():,}")

    log("2/6  Leakage-safe grouped split (by trip_uuid) ...")
    train, test = D.grouped_split(df)
    # sanity: no trip appears in both sides
    overlap = set(train[C.GROUP_KEY]) & set(test[C.GROUP_KEY])
    assert not overlap, "LEAKAGE: trips span train and test!"
    log(f"     train legs={len(train):,}  test legs={len(test):,}  trip overlap={len(overlap)}")

    log("3/6  Building corridor graph + centrality (TRAIN ONLY) ...")
    G = Gm.build_graph(train)
    cent = Gm.centrality_table(G)
    train = Gm.attach_centrality(train, cent)
    test = Gm.attach_centrality(test, cent)
    log(f"     graph nodes={G.number_of_nodes():,}  edges={G.number_of_edges():,}")

    log("4/6  Training from-scratch GraphSAGE (TRAIN graph) ...")
    emb, dim = S.train_sage(G)
    train, sage_cols = S.attach_sage(train, emb, dim)
    test, _ = S.attach_sage(test, emb, dim)

    ytr, yte = train[C.TARGET].values, test[C.TARGET].values

    feat_sets = {
        "1_baseline": C.BASELINE_FEATURES,
        "2_centrality": C.BASELINE_FEATURES + C.CENTRALITY_FEATURES,
        "3_graphsage": C.BASELINE_FEATURES + C.CENTRALITY_FEATURES + sage_cols,
    }

    log("5/6  Benchmarking ...")
    results = []
    # 0. OSRM as-is
    results.append({"model": "0_osrm_as_is", "n_features": 1,
                    **score(yte, test[C.OSRM_REFERENCE].values)})
    log(f"     OSRM as-is        MAE={results[-1]['MAE']:.2f}  within15={results[-1]['within_15pct']:.1f}%")

    models = {}
    for name, feats in feat_sets.items():
        m = _fit_xgb(train[feats], ytr)
        pred = m.predict(test[feats])
        sc = score(yte, pred)
        results.append({"model": name, "n_features": len(feats), **sc})
        models[name] = (m, feats)
        log(f"     {name:<16} MAE={sc['MAE']:.2f}  RMSE={sc['RMSE']:.2f}  "
            f"R2={sc['R2']:.3f}  within15={sc['within_15pct']:.1f}%")

    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "model_benchmark.csv", index=False)

    # graph advantage vs baseline
    base = res_df.set_index("model").loc["1_baseline"]
    sage = res_df.set_index("model").loc["3_graphsage"]
    advantage = {
        "baseline_mae": round(base.MAE, 3),
        "graphsage_mae": round(sage.MAE, 3),
        "mae_reduction_pct": round((base.MAE - sage.MAE) / base.MAE * 100, 2),
        "baseline_within15": round(base.within_15pct, 2),
        "graphsage_within15": round(sage.within_15pct, 2),
        "within15_gain_pp": round(sage.within_15pct - base.within_15pct, 2),
        "n_test_legs": int(len(test)),
    }
    (out / "graph_advantage.json").write_text(json.dumps(advantage, indent=2))

    log("6/6  Saving artifacts ...")
    best_model, best_feats = models["3_graphsage"]
    joblib.dump(best_model, out / "eta_model.joblib")
    joblib.dump({"emb": emb, "dim": dim}, out / "sage_embeddings.joblib")
    cent.to_csv(out / "node_centrality.csv", index=False)
    Gm.top_bottleneck_hubs(train, cent, k=5).to_csv(out / "top5_bottleneck_hubs.csv", index=False)
    joblib.dump({"features": best_feats, "centrality": cent}, out / "feature_meta.joblib")

    # corridor audit: chronically delayed corridors (>20% over OSRM) on train
    corr = (train.groupby([C.SOURCE, C.DEST])
            .agg(median_delay=("seg_delay_ratio", "median"),
                 trips=("seg_delay_ratio", "size")).reset_index())
    corr = corr[(corr["median_delay"] > 1.20) & (corr["trips"] >= 20)]
    corr.sort_values("median_delay", ascending=False).to_csv(
        out / "chronic_corridors.csv", index=False)

    log("\nDONE. Key result:")
    log(json.dumps(advantage, indent=2))
    return res_df, advantage


if __name__ == "__main__":
    run()
