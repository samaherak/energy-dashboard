import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="University Energy", layout="wide")
st.title("University Buildings - Energy Forecasting and Quota System")
st.caption("Dataset: BDG2 - Panther site (education buildings). "
           "Readings before June 2016 were mostly missing and are excluded. "
           "Training: Jun 2016 - May 2017. Testing: Jun - Dec 2017.")

ALL = "ALL BUILDINGS (whole campus)"
TRAIN_END = "2017-05-31"
TEST_START = "2017-06-01"
COLS = ["hour", "dayofweek", "month", "lag_24h", "lag_168h"]

@st.cache_data
def load():
    return pd.read_csv("panther_data_v2.csv", index_col=0, parse_dates=True)

def get_series(name):
    df = load()
    if name == ALL:
        return df.sum(axis=1, min_count=df.shape[1])
    return df[name]

@st.cache_data
def run_model(name):
    d = pd.DataFrame({"kwh": get_series(name)})
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["lag_24h"] = d["kwh"].shift(24)
    d["lag_168h"] = d["kwh"].shift(168)
    d = d.dropna()
    train = d.loc[:TRAIN_END]
    test = d.loc[TEST_START:]
    m = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    m.fit(train[COLS], train["kwh"])
    pred = pd.Series(m.predict(test[COLS]), index=test.index)
    mae = mean_absolute_error(test["kwh"], pred)
    naive = mean_absolute_error(test["kwh"], test["lag_24h"])
    return mae, naive, test["kwh"], pred

df = load()
name = st.sidebar.selectbox("Choose a building", [ALL] + list(df.columns))
s = get_series(name).dropna()

# ---- 1) Consumption summary ----
c1, c2, c3 = st.columns(3)
c1.metric("Average (kWh/hour)", f"{s.mean():,.1f}")
c2.metric("Peak (kWh/hour)", f"{s.max():,.1f}")
c3.metric("Total (kWh)", f"{s.sum():,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Average by hour of day")
    st.bar_chart(s.groupby(s.index.hour).mean())
with right:
    st.subheader("Average by day (0=Mon ... 6=Sun)")
    st.bar_chart(s.groupby(s.index.dayofweek).mean())

# ---- 2) Prediction ----
st.subheader("Prediction (Random Forest)")
mae, naive, actual, pred = run_model(name)
m1, m2 = st.columns(2)
m1.metric("Model error (MAE, kWh)", f"{mae:,.1f}")
m2.metric("Copy-yesterday error (MAE, kWh)", f"{naive:,.1f}")
if mae > naive:
    st.info("For this selection the model is not better than copying yesterday. "
            "Choose the whole campus for the most accurate forecast.")
st.line_chart(pd.DataFrame({"Actual": actual[:168], "Predicted": pred[:168]}))

# ---- 3) Quota ----
st.subheader("Energy quota (test period: Jun - Dec 2017)")
st.write("Quota = expected consumption minus the target reduction.")
pct = st.slider("Target reduction (%)", 0, 30, 10)
quota = pred * (1 - pct / 100)

q1, q2, q3 = st.columns(3)
q1.metric("Expected (kWh)", f"{pred.sum():,.0f}")
q2.metric("Quota (kWh)", f"{quota.sum():,.0f}")
q3.metric("Saving if quota is met (kWh)", f"{pred.sum() - quota.sum():,.0f}")

monthly = pd.DataFrame({"Actual": actual, "Quota": quota}).resample("MS").sum()
st.write("Monthly: actual vs quota")
st.line_chart(monthly)

# ---- 4) Ranking of all buildings ----
st.subheader("All buildings: quota and saving potential")
with st.spinner("Training a model for every building (first load only, about a minute)..."):
    rows = []
    for b in df.columns:
        b_mae, b_naive, b_actual, b_pred = run_model(b)
        rows.append({
            "Building": b,
            "Avg (kWh/h)": round(df[b].mean(), 1),
            "Expected (kWh)": round(b_pred.sum()),
            "Quota (kWh)": round(b_pred.sum() * (1 - pct / 100)),
            "Saving (kWh)": round(b_pred.sum() * pct / 100),
            "Model MAE": round(b_mae, 1),
            "Copy-yesterday MAE": round(b_naive, 1),
        })
table = pd.DataFrame(rows).sort_values("Saving (kWh)", ascending=False)
st.dataframe(table, use_container_width=True, hide_index=True)

st.caption("The reduction percentage is a target you choose, not a measured value. "
           "The system reports and advises; it does not control the buildings.")
