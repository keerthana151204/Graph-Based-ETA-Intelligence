# Delhivery — Graph-Based ETA Intelligence

Graph-aware delivery-ETA prediction on Delhivery's real logistics network, built
to a single standard: **every number is defensible under interview-grade
scrutiny.** The headline isn't a flashy accuracy figure — it's a *clean,
leakage-controlled* measurement of what graph structure is actually worth.

---

## TL;DR results

OSRM (the routing engine in production today) underestimates leg delivery time
badly. We model the network as a directed corridor graph and measure how much a
graph signal improves a like-for-like ETA model — on a **leakage-safe split**.

| Model | Features | MAE (min) | RMSE | R² | within 15% |
|---|---|---|---|---|---|
| OSRM as-is (today) | 1 | 17.02 | 27.20 | −0.18 | 9.2% |
| Baseline XGBoost | 11 | 11.14 | 19.39 | 0.40 | 27.5% |
| **+ Centrality** (graph) | 21 | **10.09** | 17.84 | 0.49 | **32.1%** |
| + GraphSAGE (graph) | 53 | 10.03 | 17.75 | 0.50 | 32.3% |

**Measured graph advantage (vs baseline): −10.0% MAE, +4.8pp within-15%**, on
26,932 held-out legs.

**The honest finding:** the graph signal is real and worth ~10% MAE — but it is
*almost entirely captured by cheap classical centrality features*. The
from-scratch GraphSAGE embeddings add <1% on top. So the production
recommendation is to ship the centrality model and treat the GNN as a research
baseline, not the deployed system. That conclusion is the deliverable.

![benchmark](outputs/model_benchmark.png)

---

## Why this is built the way it is

Three design choices separate this from a notebook that scores well by accident.

**1. Leakage-safe grouped split.** Each row is one *leg* of a trip; many legs
share a `trip_uuid`. A naive random split puts legs of the same trip on both
sides and inflates the score. We split on `trip_uuid`
(`GroupShuffleSplit`) and assert zero trip overlap. This is the single most
important methodological guard in the project.

**2. No outcome-derived feature, ever.** `segment_factor`, `factor`,
`actual_time`, `start_scan_to_end_scan` and friends are blocklisted in
`src/config.py::LEAKAGE_COLS`. Models that include `delay_ratio` *and*
`osrm_time` while predicting `actual_time` can hit 98–99% "accuracy" — because
`actual = ratio × osrm` reconstructs the target. We never let that happen, which
is exactly why our within-15% sits at a realistic 32%, not a suspicious 98%.

**3. Graph statistics fit on train only.** The corridor graph, centrality
metrics, and GraphSAGE embeddings are all computed from training legs, then
*mapped* onto train and test, with a train-median fallback for facilities unseen
in training. Building the graph on the full dataframe would quietly leak
test-set structure into the features.

---

## What's inside

```
src/
  config.py      single source of truth: target, leakage blocklist, hyperparams
  data.py        clean + a-priori features + grouped split
  graph.py       directed corridor graph, centrality, top-hub ranking (train only)
  graphsage.py   from-scratch GraphSAGE (mean aggregator, manual backprop, NumPy)
  features.py    -> folded into graph.py / graphsage.py attach_* helpers
  evaluate.py    MAE / RMSE / R² / within-k% (the business metric)
  train.py       the 4-way benchmark + artifact export
  plot.py        benchmark figure
  inference.py   production ETAPredictor: request -> ETA + risk band
app/
  streamlit_app.py   ops dashboard (live ETA, bottleneck hubs, benchmark)
reports/
  strategy_memo.md   business memo for the Head of Network Operations
outputs/             generated artifacts + evidence (CSV, JSON, PNG, models)
run_pipeline.py      end-to-end entrypoint
```

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py            # ~1–2 min on CPU; writes everything to outputs/
streamlit run app/streamlit_app.py
```

## GraphSAGE note (deliberate)

`src/graphsage.py` is a from-scratch NumPy implementation of the mean-aggregator
GraphSAGE (Hamilton et al., 2017):
`h_v^k = ReLU(W_k · concat(h_v^{k-1}, mean_{u∈N(v)} h_u^{k-1}))`, trained as a
supervised node regression on each hub's mean outgoing delay ratio (train graph
only), with the penultimate activations used as node embeddings. It's dependency
-light so the benchmark reproduces anywhere. A `torch_geometric` `SAGEConv`
version is a drop-in swap (see commented deps in `requirements.txt`); on this
dataset it lands in the same place — the structural signal is shallow, so a deep
GNN doesn't out-earn hand-crafted centrality. Reporting that honestly is the
point.

## Data

Delhivery operations dataset (~145k trip legs, 14.7k trips, 1.65k facilities).
Place `delivery_data.csv` in `data/`. The raw CSV is git-ignored; small result
artifacts are kept in `outputs/` as evidence.
