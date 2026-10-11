import streamlit as st
import pandas as pd

st.set_page_config(page_title="University Energy", layout="wide")
st.title("University Energy: Forecasting and Quota System")
st.caption(
    "Dataset: BDG2, site Panther (103 buildings with electricity meters and enough data, all types). "
    "Readings before June 2016 were mostly missing and are excluded. "
    "Model trained on Jun 2016 - May 2017 and tested on Jun - Dec 2017."
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
buildings_table = table[table["building"] != ALL].copy()


def series_of(name):
    return campus if name == ALL else df[name]


# ---------------- sidebar: who is using the system ----------------
role = st.sidebar.radio("View as", ["University administration", "Building manager"])
pct = st.sidebar.slider("Target reduction (%)", 0, 30, 10)
st.sidebar.caption("The reduction is a target you choose, not a measured value.")


# ================= ADMINISTRATION PANEL =================
def admin_panel():
    st.header("University administration panel")
    st.write("Whole site: all buildings, quotas and where the saving potential is.")

    pred = preds[ALL].dropna()
    actual = campus.reindex(pred.index)
    quota = pred * (1 - pct / 100)

    a1, a2, a3 = st.columns(3)
    a1.metric("Expected (kWh/hour)", f"{pred.mean():,.0f}")
    a2.metric("Quota (kWh/hour)", f"{quota.mean():,.0f}")
    a3.metric("Saving per 30-day month (kWh)", f"{(pred.mean() - quota.mean()) * 24 * 30:,.0f}")

    st.write("Monthly average for the whole site: actual vs quota (kWh/hour)")
    st.line_chart(pd.DataFrame({"Actual": actual, "Quota": quota}).resample("MS").mean())

    tbl = buildings_table.copy()
    types = sorted(tbl["type"].unique())
    chosen = st.multiselect("Building types", types, default=types)
    only_reliable = st.checkbox("Show only buildings with a reliable forecast", value=False)
    tbl = tbl[tbl["type"].isin(chosen)]
    if only_reliable:
        tbl = tbl[tbl["reliable"]]
    tbl["quota_kwh"] = (tbl["expected_kwh"] * (1 - pct / 100)).round()
    tbl["saving_kwh"] = (tbl["expected_kwh"] * pct / 100).round()
    tbl = tbl.sort_values("saving_kwh", ascending=False)

    st.write("Top 10 buildings by saving potential over the test period (kWh)")
    st.bar_chart(tbl.head(10).set_index("building")["saving_kwh"])

    st.write("All buildings: quota and saving potential")
    show = tbl[["building", "type", "avg_kwh_h", "expected_kwh", "quota_kwh",
                "saving_kwh", "model_mae", "naive_mae", "reliable"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.write("Forecast reliability by building type")
    summary = buildings_table.groupby("type").agg(
        buildings=("building", "count"), reliable=("reliable", "sum"))
    st.dataframe(summary)
    st.caption("The system reports and advises; it does not control the buildings.")


# ================= BUILDING MANAGER PANEL =================
def manager_panel():
    st.header("Building manager panel")
    name = st.selectbox("Choose your building", list(buildings_table["building"]))
    row = buildings_table[buildings_table["building"] == name].iloc[0]
    st.write("Type: " + str(row["type"]))

    s = series_of(name).dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average (kWh/hour)", f"{s.mean():,.1f}")
    c2.metric("Peak (kWh/hour)", f"{s.max():,.1f}")
    c3.metric("Hours with data", f"{len(s):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Average by hour of day")
        st.bar_chart(s.groupby(s.index.hour).mean())
    with right:
        st.subheader("Average by day (0=Mon ... 6=Sun)")
        st.bar_chart(s.groupby(s.index.dayofweek).mean())

    pred = preds[name].dropna()
    actual = series_of(name).reindex(pred.index)
    quota = pred * (1 - pct / 100)

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

    st.subheader("Your quota")
    q1, q2, q3 = st.columns(3)
    q1.metric("Expected (kWh/hour)", f"{pred.mean():,.1f}")
    q2.metric("Quota (kWh/hour)", f"{quota.mean():,.1f}")
    both = pd.DataFrame({"a": actual, "q": quota}).dropna()
    above = (both["a"] > both["q"]).mean() * 100 if len(both) else 0
    q3.metric("Hours above quota (%)", f"{above:,.0f}%")
    st.caption("Hours above quota = share of hours where the real consumption was higher "
               "than the quota (calculated on the test period).")

    st.write("Monthly average: actual vs quota (kWh/hour)")
    st.line_chart(pd.DataFrame({"Actual": actual, "Quota": quota}).resample("MS").mean())


if role == "University administration":
    admin_panel()
else:
    manager_panel()
