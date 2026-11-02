"""Field Force Tracker - Streamlit version.

Run:  streamlit run app.py
Reads roster.csv and visits.csv from the same folder (create them with: python gen_data.py).

Metrics are ratios of sums:
  attendance = reps present / reps rostered
  coverage   = visits completed / visits planned
  OSA        = SKUs available / SKUs audited
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
TARGETS = {"att": 92.0, "cov": 90.0, "osa": 88.0, "vpd": 6.0}
COLORS = {"att": "#2a78d6", "cov": "#eb6834", "osa": "#1baf7a"}
NAMES = {"att": "Attendance", "cov": "Coverage", "osa": "On-shelf availability"}

st.set_page_config(page_title="Field Force Tracker", layout="wide")


# ---------- data ----------
@st.cache_data
def load():
    roster = pd.read_csv(HERE / "roster.csv", parse_dates=["date"])
    visits = pd.read_csv(HERE / "visits.csv", parse_dates=["date"])
    return roster, visits


def daily(roster: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    d = roster.groupby("date").agg(present=("present", "sum"), rostered=("present", "size"))
    return d.join(visits.groupby("date").agg(
        planned=("planned", "sum"), completed=("completed", "sum"),
        audited=("skus_audited", "sum"), avail=("skus_available", "sum")))


def ratios(d: pd.DataFrame) -> dict:
    return {
        "att": 100 * d.present.sum() / d.rostered.sum(),
        "cov": 100 * d.completed.sum() / d.planned.sum(),
        "osa": 100 * d.avail.sum() / d.audited.sum(),
        "vpd": d.completed.sum() / d.present.sum(),
    }


def status(att: float, cov: float, osa: float) -> str:
    misses = int(att < TARGETS["att"]) + int(cov < TARGETS["cov"]) + int(osa < TARGETS["osa"])
    return ["On track", "Watch", "At risk", "At risk"][misses]


if not (HERE / "roster.csv").exists() or not (HERE / "visits.csv").exists():
    st.error("roster.csv and visits.csv not found. Run `python gen_data.py` in this folder first.")
    st.stop()

roster_all, visits_all = load()
all_regions = sorted(roster_all.region.unique())
min_d, max_d = roster_all.date.min().date(), roster_all.date.max().date()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
roll = st.sidebar.slider("Trend smoothing (rolling working days)", 1, 7, 3)
st.sidebar.caption("Targets: attendance 92%, coverage 90%, OSA 88%, 6.0 visits per rep-day. Edit TARGETS at the top of app.py.")

if not regions:
    st.warning("Select at least one region.")
    st.stop()
if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.info("Pick an end date to complete the range.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
r = roster_all[roster_all.region.isin(regions) & roster_all.date.between(start, end)]
v = visits_all[visits_all.region.isin(regions) & visits_all.date.between(start, end)]
d = daily(r, v)

# ---------- header + KPIs ----------
st.title("Field Force Tracker")
st.caption(f"{start:%d %b} to {end:%d %b %Y} · {len(d)} working days · {', '.join(regions)} · sample data")

now = ratios(d)
delta = {k: None for k in now}
if len(d) >= 14:
    last7, prev7 = ratios(d.iloc[-7:]), ratios(d.iloc[-14:-7])
    delta = {k: last7[k] - prev7[k] for k in now}

cols = st.columns(4)
specs = [
    ("att", "Attendance", f"{now['att']:.1f}%", "pp"),
    ("cov", "Beat coverage", f"{now['cov']:.1f}%", "pp"),
    ("osa", "On-shelf availability", f"{now['osa']:.1f}%", "pp"),
    ("vpd", "Visits per rep-day", f"{now['vpd']:.1f}", "visits"),
]
for col, (key, label, val, unit) in zip(cols, specs):
    dtxt = None if delta[key] is None else f"{delta[key]:+.1f} {unit} vs prior 7 days"
    col.metric(label, val, dtxt)
    tgt = TARGETS[key]
    ok = now[key] >= tgt
    col.caption(f"Target {tgt:g}{'%' if key != 'vpd' else ''} · {'meets target' if ok else 'below target'}")

# ---------- trend ----------
st.subheader("Daily trend")
rs = d.rolling(roll).sum().dropna()
trend = pd.DataFrame({
    "att": 100 * rs.present / rs.rostered,
    "cov": 100 * rs.completed / rs.planned,
    "osa": 100 * rs.avail / rs.audited,
})
fig = go.Figure()
for key in ("att", "cov", "osa"):
    fig.add_trace(go.Scatter(x=trend.index, y=trend[key], name=NAMES[key], mode="lines",
                             line=dict(color=COLORS[key], width=2.5),
                             hovertemplate="%{y:.1f}%"))
fig.update_layout(
    height=380, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(ticksuffix="%", title=None), xaxis=dict(title=None, tickformat="%d %b"),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(fig, width="stretch")

# ---------- scorecard + weakest stores ----------
left, right = st.columns([1.25, 1])

with left:
    st.subheader("Region scorecard")
    rows = []
    for reg in regions:
        k = ratios(daily(r[r.region == reg], v[v.region == reg]))
        rows.append({
            "Region": reg,
            "Reps": r[r.region == reg].rep.nunique(),
            "Attendance %": k["att"], "Coverage %": k["cov"], "OSA %": k["osa"],
            "Status": status(k["att"], k["cov"], k["osa"]),
        })
    sc = pd.DataFrame(rows)
    st.dataframe(
        sc, hide_index=True, width="stretch",
        column_config={
            "Attendance %": st.column_config.NumberColumn(format="%.1f%%", help="Target 92%"),
            "Coverage %": st.column_config.NumberColumn(format="%.1f%%", help="Target 90%"),
            "OSA %": st.column_config.NumberColumn(format="%.1f%%", help="Target 88%"),
        })
    st.download_button("Download scorecard (CSV)", sc.to_csv(index=False).encode("utf-8"),
                       file_name="region_scorecard.csv", mime="text/csv")

with right:
    st.subheader("Lowest availability stores")
    g = v.groupby(["store", "region"]).agg(audited=("skus_audited", "sum"), avail=("skus_available", "sum"),
                                           visits=("completed", "sum")).reset_index()
    g = g[g.audited > 0]
    g["OSA %"] = 100 * g.avail / g.audited
    g["vs target (pp)"] = g["OSA %"] - TARGETS["osa"]
    low = g.sort_values("OSA %").head(8)[["store", "region", "OSA %", "vs target (pp)", "visits"]]
    low.columns = ["Store", "Region", "OSA %", "vs target (pp)", "Visits"]
    st.dataframe(
        low, hide_index=True, width="stretch",
        column_config={
            "OSA %": st.column_config.NumberColumn(format="%.1f%%"),
            "vs target (pp)": st.column_config.NumberColumn(format="%+.1f"),
        })

st.caption("Each figure is a ratio of totals, not an average of daily percentages. "
           "Arrows compare the last 7 working days in the selected range with the 7 before.")
