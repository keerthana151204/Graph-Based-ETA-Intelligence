"""
Production inference.

Loads the trained model + graph artifacts and turns a single dispatch request
into an ETA prediction, an OSRM-divergence read, and an operational risk band.
This is the layer an ops dashboard or routing service calls — it never needs the
training data or any outcome columns.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import joblib
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class ETARequest:
    source_center: str
    destination_center: str
    segment_osrm_time: float
    segment_osrm_distance: float
    osrm_time: float
    osrm_distance: float
    route_type: str = "Carting"     # or "FTL"
    hour: int = 12
    day_of_week: int = 2
    is_cutoff: int = 0
    cutoff_factor: float = 0.0


class ETAPredictor:
    def __init__(self, output_dir=C.OUTPUT_DIR):
        self.model = joblib.load(output_dir / "eta_model.joblib")
        meta = joblib.load(output_dir / "feature_meta.joblib")
        self.features = meta["features"]
        self.cent = meta["centrality"].set_index("node")
        sage = joblib.load(output_dir / "sage_embeddings.joblib")
        self.emb, self.dim = sage["emb"], sage["dim"]
        self._cent_fallback = self.cent.median(numeric_only=True)

    def _row(self, req: ETARequest) -> pd.DataFrame:
        d = {
            "segment_osrm_time": req.segment_osrm_time,
            "segment_osrm_distance": req.segment_osrm_distance,
            "osrm_time": req.osrm_time,
            "osrm_distance": req.osrm_distance,
            "route_type_enc": int(str(req.route_type).upper() == "FTL"),
            "hour": req.hour,
            "day_of_week": req.day_of_week,
            "is_weekend": int(req.day_of_week >= 5),
            "is_peak": int(req.hour in (8, 9, 10, 17, 18, 19, 20)),
            "is_cutoff": req.is_cutoff,
            "cutoff_factor": req.cutoff_factor,
        }
        for who, node in (("src", req.source_center), ("dst", req.destination_center)):
            c = self.cent.loc[node] if node in self.cent.index else self._cent_fallback
            for k in ["betweenness", "pagerank", "in_degree", "out_degree", "clustering"]:
                d[f"{who}_{k}"] = float(c[k])
            e = self.emb.get(node, np.zeros(self.dim))
            for i in range(self.dim):
                d[f"{who}_sage_{i}"] = float(e[i])
        return pd.DataFrame([d])[self.features]

    def predict(self, req: ETARequest) -> dict:
        x = self._row(req)
        eta = float(self.model.predict(x)[0])
        osrm = req.segment_osrm_time
        divergence = (eta - osrm) / max(osrm, 1e-6)
        if divergence < 0.15:
            risk = "LOW"
        elif divergence < 0.40:
            risk = "MEDIUM"
        elif divergence < 0.80:
            risk = "HIGH"
        else:
            risk = "CRITICAL"
        return {
            "predicted_eta_min": round(eta, 1),
            "osrm_eta_min": round(osrm, 1),
            "divergence_pct": round(divergence * 100, 1),
            "risk_band": risk,
            "request": asdict(req),
        }


if __name__ == "__main__":
    p = ETAPredictor()
    demo = ETARequest(
        source_center="IND000000ACB", destination_center="IND562132AAA",
        segment_osrm_time=120, segment_osrm_distance=210,
        osrm_time=120, osrm_distance=210, route_type="Carting", hour=18,
    )
    import json
    print(json.dumps(p.predict(demo), indent=2))
