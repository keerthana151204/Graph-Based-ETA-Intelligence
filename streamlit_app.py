"""
Network-operations dashboard (Streamlit).

Run:  streamlit run app/streamlit_app.py
A non-technical ops user can predict an ETA, see the risk band, and inspect the
top bottleneck hubs and chronic corridors — without touching the model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.inference import ETAPredictor, ETARequest          # noqa: E402
from src import config as C                                  # noqa: E402

st.set_page_config(page_title="Delhivery Graph-ETA", layout="wide")
st.title("Delhivery — Graph-Based ETA Intelligence")
st.caption("Leakage-safe ETA prediction · bottleneck hubs · corridor risk")


@st.cache_resource
def get_predictor():
    return ETAPredictor()


@st.cache_data
def load(name):
    return pd.read_csv(C.OUTPUT_DIR / name)


tab1, tab2, tab3 = st.tabs(["Live ETA", "Bottleneck hubs", "Model benchmark"])

with tab1:
    st.subheader("Predict a leg ETA")
    c1, c2, c3 = st.columns(3)
    src = c1.text_input("Source center", "IND000000ACB")
    dst = c2.text_input("Destination center", "IND562132AAA")
    rt = c3.selectbox("Route type", ["Carting", "FTL"])
    osrm_t = c1.number_input("OSRM time (min)", value=120.0, min_value=1.0)
    osrm_d = c2.number_input("OSRM distance (km)", value=210.0, min_value=0.1)
    hour = c3.slider("Dispatch hour", 0, 23, 18)
    if st.button("Predict ETA", type="primary"):
        req = ETARequest(source_center=src, destination_center=dst,
                         segment_osrm_time=osrm_t, segment_osrm_distance=osrm_d,
                         osrm_time=osrm_t, osrm_distance=osrm_d, route_type=rt, hour=hour)
        r = get_predictor().predict(req)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted ETA", f"{r['predicted_eta_min']} min")
        m2.metric("OSRM ETA", f"{r['osrm_eta_min']} min", f"{r['divergence_pct']}%")
        m3.metric("Risk band", r["risk_band"])

with tab2:
    st.subheader("Top 5 bottleneck hubs")
    st.dataframe(load("top5_bottleneck_hubs.csv"), use_container_width=True)
    st.subheader("Chronically delayed corridors (>20% over OSRM)")
    st.dataframe(load("chronic_corridors.csv").head(25), use_container_width=True)

with tab3:
    st.subheader("Model benchmark (grouped split by trip_uuid)")
    st.dataframe(load("model_benchmark.csv"), use_container_width=True)
    img = C.OUTPUT_DIR / "model_benchmark.png"
    if img.exists():
        st.image(str(img))
