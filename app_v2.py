import streamlit as st
import pandas as pd

st.set_page_config(page_title="University Energy", layout="wide")
st.title("University Energy: Forecasting and Quota System")
st.caption(
    "Dataset: BDG2, site Panther (103 buildings with electricity meters and enough data, all types). "
    "Readings before June 2016 were mostly missing and are excluded. "
    "Model trained on Jun 2016 - May 2017 and tested on Jun - Dec 2017 (about 7 months)."
)

ALL = "ALL_BUILDINGS"

@st.cache_data
def load():
    df = pd.read_csv("panther_all_v3.csv", index_col=0, parse_dates=True)
    preds = pd.read_csv("preds_v3.csv", index_col=0, parse_dates=True)
    table = pd.read_csv("building_table_v3.csv")
    return df, preds, table

df, preds, table = load()
campus = df.sum(axis=1, min_count=df.shape[1])

def series_of(name):
    return campus if name == ALL else df[name]

names = [ALL] + list(df.columns)
name = st.sidebar.selectbox(
    "Choose a building", names,
    format_func=lambda x: "ALL BUILDINGS (whole site)" if x == ALL else x)
row = table[table["building"] == name].iloc[0]
st.sidebar.write("Type: " + str(row["type"]))
s = series_of(name).dropna()

# ---- 1) Consumption ----
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

# ---- 2) Forecast ----
pred = preds[name].dropna()
actual = series_of(name).reindex(pred.index)

st.subheader("Forecast (test period: Jun - Dec 2017)")
m1, m2 = st.columns(2)
m1.metric("Model error (MAE, kWh)", f"{row['model_mae']:,.1f}")
m2.metric("Copy-yesterday error (MAE, kWh)", f"{row['naive_mae']:,.1f}")
if row["reliable"]:
    st.success("Forecast is reliable here: the model beat copying yesterday.")
else:
    st.warning("Forecast is NOT reliable here: copying yesterday was more accurate "
               "than the model. Treat the quota below as indicative only.")
st.line_chart(pd.DataFrame({"Actual": actual.iloc[:168], "Predicted": pred.iloc[:168]}))

# ---- 3) Quota ----
st.subheader("Energy quota")
st.write("Quota = expected consumption minus the target reduction.")
pct = st.slider("Target reduction (%)", 0, 30, 10)
expected = pred.sum()
quota = pred * (1 - pct / 100)

q1, q2, q3 = st.columns(3)
q1.metric("Expected (kWh)", f"{expected:,.0f}")
q2.metric("Quota (kWh)", f"{quota.sum():,.0f}")
q3.metric("Saving if quota is met (kWh)", f"{expected - quota.sum():,.0f}")

st.write("Monthly: actual vs quota")
st.line_chart(pd.DataFrame({"Actual": actual, "Quota": quota}).resample("MS").sum())

# ---- 4) All buildings ----
st.subheader("All buildings: quota and saving potential")
tbl = table[table["building"] != ALL].copy()
types = sorted(tbl["type"].unique())
chosen = st.multiselect("Building types", types, default=types)
only_reliable = st.checkbox("Show only buildings with a reliable forecast", value=False)
tbl = tbl[tbl["type"].isin(chosen)]
if only_reliable:
    tbl = tbl[tbl["reliable"]]
tbl["quota_kwh"] = (tbl["expected_kwh"] * (1 - pct / 100)).round()
tbl["saving_kwh"] = (tbl["expected_kwh"] * pct / 100).round()
tbl = tbl.sort_values("saving_kwh", ascending=False)

st.write("Top 10 buildings by saving potential (kWh)")
st.bar_chart(tbl.head(10).set_index("building")["saving_kwh"])

show = tbl[["building", "type", "avg_kwh_h", "expected_kwh", "quota_kwh",
            "saving_kwh", "model_mae", "naive_mae", "reliable"]]
st.dataframe(show, use_container_width=True, hide_index=True)

st.write("Forecast reliability by building type")
summary = table[table["building"] != ALL].groupby("type").agg(
    buildings=("building", "count"), reliable=("reliable", "sum"))
st.dataframe(summary)

st.caption("The reduction percentage is a target you choose, not a measured value. "
           "The system reports and advises; it does not control the buildings.")
