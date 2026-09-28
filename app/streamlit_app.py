# Forewell: live hydrate early-warning dashboard.
#
# Operator-facing layout. Controls are visible by default. A large status
# banner reads at a glance, with a short instruction so a first-time
# visitor knows exactly what to do. The sensor chart uses the full width.

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Forewell - Hydrate Early Warning",
                   page_icon="🛢", layout="wide",
                   initial_sidebar_state="expanded")

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_DIR / "forewell_model.joblib")


@st.cache_data
def load_demo():
    return pd.read_parquet(ARTIFACT_DIR / "demo_wells.parquet")


art = load_artifact()
model = art["model"]
feature_cols = art["feature_cols"]
q_hat = art["conformal_q"]
alpha = art["alpha"]
demo = load_demo()

# --- Header ---
st.title("Forewell: hydrate early warning for offshore wells")
st.markdown(
    "Forewell reads a well's sensor data and raises an alarm while a hydrate "
    "is still forming, during the window when an operator can still act. It "
    "was trained and tested on real wells from the Petrobras 3W dataset. Each "
    "alarm carries a calibrated confidence level and the sensor readings "
    "behind it."
)
st.info(
    "**How to use:** in the left panel, pick a well, then drag the **Time "
    "position** slider. The status banner below tracks the well second by "
    "second, from normal operation into a forming hydrate.",
    icon="👈",
)

# --- Sidebar controls ---
st.sidebar.header("Controls")
wells = sorted(demo["well"].unique())
well = st.sidebar.selectbox("1. Select a well", wells)
well_df = demo[demo["well"] == well].sort_values("win_end_idx").reset_index(drop=True)
pos = st.sidebar.slider("2. Time position (drag me)", 0, len(well_df) - 1, 0)
hours_in = well_df.iloc[pos]["win_end_idx"] / 3600.0
st.sidebar.metric("Time into recording", f"{hours_in:.1f} hours")
st.sidebar.divider()
st.sidebar.markdown(
    f"**Confidence level:** {int((1 - alpha) * 100)}%\n\n"
    "Forewell uses conformal prediction. On readings where the evidence is "
    "thin, it returns **Uncertain** and leaves the call to the operator."
)

# --- Score the current window ---
row = well_df.iloc[pos]
x = np.nan_to_num(row[feature_cols].to_numpy(dtype=float)).reshape(1, -1)
proba = model.predict_proba(x)[0]
score = float(proba[1])
true_label = int(row["label"])

in_set = {c: (1.0 - proba[c]) <= q_hat for c in (0, 1)}
if in_set[1] and not in_set[0]:
    state, banner = "DEVELOPING HYDRATE", "error"
elif in_set[0] and not in_set[1]:
    state, banner = "NORMAL", "success"
else:
    state, banner = "UNCERTAIN, refer to operator", "warning"

st.divider()

# --- Status banner, full width ---
if banner == "error":
    st.error(f"# 🛑 {state}")
elif banner == "success":
    st.success(f"# ✅ {state}")
else:
    st.warning(f"# ⚠️ {state}")

m1, m2, m3 = st.columns(3)
m1.metric("Detector score", f"{score:.2f}",
          help="Model probability that a hydrate is developing (0 to 1).")
m2.metric("System confidence", f"{int((1 - alpha) * 100)}%")
m3.metric("Recorded state",
          "developing" if true_label == 1 else "normal",
          help="What this window was actually labelled in the dataset. Shown so you can check the alarm against the record.")

st.divider()

# --- Sensor history, full width ---
st.subheader(f"Sensor history: {well}")
st.caption("Sensor readings up to the selected time. 'Drift' is how far a "
           "sensor has moved from this well's own baseline. As a hydrate "
           "forms, temperature tends to fall and drift tends to grow.")
shown = well_df.iloc[: pos + 1]
sensors = [s for s in ["T-TPT_min", "QGL_drift_base", "P-PDG_drift_base",
                       "P-MON-CKP_drift_base"] if s in well_df.columns]
chart_df = shown[sensors].copy()
chart_df.index = (shown["win_end_idx"] / 3600.0).round(2)
chart_df.index.name = "hours into recording"
st.line_chart(chart_df, height=380, use_container_width=True)

st.divider()

# --- Drivers, full width ---
st.subheader("Sensors behind this reading")
st.caption("The sensor features carrying the most weight at this moment.")
vals = row[feature_cols].astype(float)
top = vals.reindex(vals.abs().sort_values(ascending=False).index).head(8)
drivers = pd.DataFrame({"sensor feature": top.index,
                        "value": top.values.round(3)})
st.dataframe(drivers, use_container_width=True, hide_index=True, height=320)

st.divider()
st.caption(
    "Forewell is a research prototype built on the public Petrobras 3W "
    "dataset (CC BY 4.0). It shows early-warning detection, calibrated "
    "confidence, and explainable alarms on real offshore well data. It is "
    "not a certified operational system."
)