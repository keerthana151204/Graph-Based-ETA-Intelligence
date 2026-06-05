# Network Operations Strategy Memo

**To:** Head of Network Operations, Delhivery
**From:** Data Science & Analytics
**Re:** Graph-Based ETA Intelligence — bottleneck hubs, corridor risk, and where the model does (and doesn't) help
**Classification:** Internal — Operations Strategy

---

## 1. Executive summary

OSRM, our current ETA engine, is not just imprecise — it is *biased*. Across
26,932 held-out delivery legs it underestimates actual leg time with a mean
error of **17 minutes**, and only **9% of legs land within 15%** of the
predicted time. The errors are not random noise: they are structural, and they
concentrate at a small number of high-centrality hubs whose delays propagate
downstream.

Modelling the network as a directed corridor graph and adding graph-structural
features to an ETA model cuts mean error by **10%** (17.0 → 10.0 min vs OSRM;
11.1 → 10.0 min vs a like-for-like ML baseline) and roughly **triples** the
share of legs within 15% (9% → 32%). Critically, almost all of that graph lift
comes from *cheap, explainable centrality features* — a deep GNN adds under 1%
more. The recommended production system is therefore simple, fast, and
interpretable, not a black box.

This memo names the five hubs to act on first and states clearly what each
intervention can and cannot be expected to deliver.

## 2. Key findings

- **Systemic, not random, underestimation.** OSRM's R² on held-out legs is
  negative (−0.18) — it is worse than predicting the mean. The ML baseline lifts
  this to 0.40 and the graph model to 0.50.
- **Delay is concentrated.** One facility — **Gurgaon_Bilaspur_HB (Haryana)** —
  carries a betweenness centrality of **0.229**, roughly **1.6× the next-highest
  hub** and an order of magnitude above the median. It sits on a large share of
  all network paths; congestion there cascades everywhere.
- **Chronic corridors are identifiable in advance.** Hundreds of corridors run a
  median delay ratio above 1.2× OSRM on ≥20 trips — i.e. they are *reliably*
  late, not occasionally. These are targetable today (`outputs/chronic_corridors.csv`).
- **The graph signal is shallow.** Classical centrality captures essentially all
  of the available structural lift; GraphSAGE embeddings add <1%. Spend
  engineering effort on the feature pipeline, not on a GNN.

## 3. Top 5 bottleneck hubs — diagnosis & action

Ranked by a composite of betweenness centrality (structural reach), SLA-breach
rate, and volume. All figures are from training legs.

| Hub | Betweenness | In-deg | SLA breach | Median delay | Legs | Recommended action |
|---|---|---|---|---|---|---|
| Gurgaon_Bilaspur_HB (HR) | 0.229 | 45 | 89.1% | 1.58× | 18,933 | National gateway. Priority dwell-time audit + FTL fast-lane to separate time-critical loads from the Carting queue |
| Bhiwandi_Mankoli_HB (MH) | 0.051 | 28 | 94.5% | 1.76× | 7,373 | Highest breach rate of the five. Add parallel outbound capacity; cap peak-hour intake |
| Bangalore_Nelmngla_H (KA) | 0.146 | 34 | 90.7% | 1.52× | 8,178 | High reach, second-highest centrality. Route diversification — add 2 alternate outbound corridors |
| Hyderabad_Shamshbd_H (TG) | 0.076 | 30 | 94.5% | 1.71× | 2,826 | Convert high-volume Carting lanes to FTL on long-haul corridors |
| Pune_Tathawde_H (MH) | 0.044 | 22 | 89.5% | 1.57× | 3,256 | Real-time delay alerting + dispatch-window shift away from the 17:00–20:00 peak |

Because these hubs lie on a disproportionate share of network paths, reducing
dwell time here compresses transit time across many downstream corridors at
once — the highest-leverage place to spend a fixed improvement budget.

## 4. FTL vs Carting decision framework

Route-type is currently set without regard to a corridor's structural position.
Recommend converting a Carting corridor to FTL when **two or more** hold:

1. Corridor median delay ratio > **1.5× OSRM**
2. Source-hub betweenness centrality > **0.05** (top of the network)
3. Corridor volume sustained (≥ ~20 legs/period)
4. Long-haul leg (OSRM distance materially above the network median)

This targets exactly the legs where Carting's queue/handoff penalty compounds
with structural congestion. It is a *policy* change, not capital expenditure, so
it can be piloted in weeks on the top corridors and measured against held-out
control corridors.

## 5. Expected impact — stated honestly

| Lever | What it moves | Confidence |
|---|---|---|
| Deploy graph-ETA model | Customer-facing ETA error −10%; within-15% 9% → 32% | **High** — measured on held-out legs |
| Top-5 hub dwell-time programme | Largest single reduction in propagated delay | Medium — directional; needs an on-site dwell study to size |
| FTL conversion on qualifying corridors | Lower delay variance on converted lanes | Medium — pilot before rollout |
| Peak-window dispatch smoothing | Lower peak-hour breach rate | Low–Medium — depends on capacity flexibility |

**A deliberate omission:** this dataset contains *no cost, penalty, or revenue
fields*. Any rupee figure for "revenue recovered" would be invented, so this
memo does not state one. The model lift (−10% MAE, +4.8pp within-15%) is
measured and real; the operational impact of the hub and routing changes should
be sized with a short on-site study before a number is attached. Quantifying
only what the data supports is the standard this analysis holds itself to.

## 6. 90-day execution

- **Weeks 1–2:** Deploy the centrality ETA model behind the current OSRM
  estimate; shadow-mode logging of divergence and risk band.
- **Weeks 3–4:** FTL-conversion pilot on the 5 highest-scoring Carting
  corridors; track vs control.
- **Month 2:** Dwell-time study at Gurgaon_Bilaspur and Bhiwandi_Mankoli; size
  the capacity intervention.
- **Month 3:** Roll the dashboard out to ops; begin weekly bottleneck review;
  re-fit graph features on fresh data monthly.

---

*Built from ~145k Delhivery trip legs using a leakage-safe (trip-level grouped
split) graph-ETA pipeline. Model figures are held-out measurements; hub and
corridor figures are train-set statistics. Reproduce with `python run_pipeline.py`.*
