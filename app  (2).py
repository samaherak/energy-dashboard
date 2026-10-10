import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="University Energy", layout="wide")
st.title("Cloud-Based Analysis of Electricity Consumption in University Buildings")
st.caption("Dataset: BDG2 - Panther site (education buildings), 2016-2017")

@st.cache_data
def load():
    return pd.read_csv("panther_data.csv", index_col=0, parse_dates=True)

@st.cache_data
def run_model(building):
    d = pd.DataFrame({"kwh": load()[building]})
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["lag_24h"] = d["kwh"].shift(24)
    d["lag_168h"] = d["kwh"].shift(168)
    d = d.dropna()
    cols = ["hour", "dayofweek", "month", "lag_24h", "lag_168h"]
    train = d[d.index.year == 2016]
    test = d[d.index.year == 2017]
    m = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    m.fit(train[cols], train["kwh"])
    pred = m.predict(test[cols])
    mae = mean_absolute_error(test["kwh"], pred)
    naive = mean_absolute_error(test["kwh"], test["lag_24h"])
    week = pd.DataFrame({"Actual": test["kwh"][:168].values,
                         "Predicted": pred[:168]}, index=test.index[:168])
    return mae, naive, week

df = load()
building = st.sidebar.selectbox("Choose a building", df.columns)
s = df[building].dropna()

c1, c2, c3 = st.columns(3)
c1.metric("Average (kWh/hour)", f"{s.mean():.1f}")
c2.metric("Peak (kWh/hour)", f"{s.max():.1f}")
c3.metric("Total (kWh)", f"{s.sum():,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Average by hour of day")
    st.bar_chart(s.groupby(s.index.hour).mean())
with right:
    st.subheader("Average by day (0=Mon ... 6=Sun)")
    st.bar_chart(s.groupby(s.index.dayofweek).mean())

st.subheader("Prediction (Random Forest, trained on 2016, tested on 2017)")
mae, naive, week = run_model(building)
m1, m2 = st.columns(2)
m1.metric("Model error (MAE, kWh)", f"{mae:.1f}")
m2.metric("Copy-yesterday error (MAE, kWh)", f"{naive:.1f}")
st.line_chart(week)

st.subheader("Savings scenario (night hours 22:00 - 05:00)")
pct = st.slider("Reduce night consumption by (%)", 0, 30, 10)
night = s[s.index.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])].mean()
saving = night * pct / 100 * 8 * 365
st.metric("Expected yearly saving (kWh)", f"{saving:,.0f}")
st.caption("The percentage is a scenario, not a measured value.")
