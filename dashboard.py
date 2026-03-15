"""
Calgary Housing Pressure Dashboard
Live daily indicator: oil price + economic signals → housing market direction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess, sys, os

st.set_page_config(
    page_title="Calgary Housing Pressure",
    page_icon="🏠",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-label { color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 2rem; font-weight: 800; margin: 4px 0; }
  .metric-delta { font-size: 0.8rem; color: #888; }
  .positive { color: #22c55e; }
  .negative { color: #ef4444; }
  .neutral  { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    """Load pre-computed pressure scores and raw indicators."""
    pressure = pd.read_csv("data/processed/daily_housing_pressure.csv", parse_dates=["date"])
    oil_daily = pd.read_csv("data/raw/oil_prices_daily.csv", parse_dates=["date"])
    creb = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    creb_tr = creb[(creb["district"] == "Calgary") & (creb["property_type"] == "Total Residential")].copy()
    creb_tr["benchmark_mom_pct"] = creb_tr["benchmark_price"].pct_change() * 100
    return pressure, oil_daily, creb_tr


def refresh_and_retrain():
    """Re-scrape latest data and rerun models."""
    with st.spinner("Fetching latest oil prices..."):
        subprocess.run([sys.executable, "src/scrape_oil.py"], cwd=os.getcwd(), capture_output=True)
    with st.spinner("Fetching latest economic indicators..."):
        subprocess.run([sys.executable, "src/scrape_daily_indicators.py"], cwd=os.getcwd(), capture_output=True)
    with st.spinner("Rebuilding merged datasets..."):
        subprocess.run([sys.executable, "src/merge_datasets.py"], cwd=os.getcwd(), capture_output=True)
    with st.spinner("Retraining daily model..."):
        subprocess.run([sys.executable, "src/model_daily.py"], cwd=os.getcwd(), capture_output=True)
    st.cache_data.clear()
    st.rerun()


# ── Load ───────────────────────────────────────────────────────────────────────
pressure, oil_daily, creb_tr = load_data()

today = pressure.iloc[-1]
prev = pressure.iloc[-2]
score_today = today["housing_pressure_score"]
score_prev  = prev["housing_pressure_score"]
oil_today   = today["oil_price_usd"]
score_delta = score_today - score_prev

# Last known CREB benchmark
last_creb   = creb_tr.iloc[-1]
last_bench  = last_creb["benchmark_price"]
last_mom    = last_creb["benchmark_mom_pct"]

# Predicted next benchmark
pred_next   = last_bench * (1 + score_today / 100)


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.title("🏠 Calgary Housing Pressure Dashboard")
    st.caption(f"Updated through {today['date'].strftime('%B %d, %Y')} · Data: CREB, Bank of Canada, WTI Oil")
with col_refresh:
    st.write("")
    st.write("")
    if st.button("⟳ Refresh Data", use_container_width=True):
        refresh_and_retrain()


# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, delta=None, fmt="", cls="neutral"):
    color_class = cls
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color_class}">{fmt}{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

score_cls = "positive" if score_today > 0 else "negative"
oil_cls   = "negative" if oil_today > 80 else ("positive" if oil_today < 60 else "neutral")

kpi(k1, "Housing Pressure Score", f"{score_today:+.2f}%", f"Δ {score_delta:+.2f}% from yesterday", cls=score_cls)
kpi(k2, "WTI Oil (today)", f"{oil_today:.2f}", "USD / barrel", fmt="$", cls=oil_cls)
kpi(k3, "Last CREB Benchmark", f"{last_bench/1000:.1f}k",
    f"MoM: {last_mom:+.1f}% ({last_creb['date'].strftime('%b %Y')})", fmt="$", cls="neutral")
kpi(k4, "Predicted Next Benchmark", f"{pred_next/1000:.1f}k", "Based on today's score", fmt="$",
    cls="positive" if pred_next > last_bench else "negative")
kpi(k5, "Rate (BoC)", f"{today['overnight_rate']:.2f}%", "Overnight rate", cls="neutral")

st.write("")

# ── Main chart: oil + pressure score ──────────────────────────────────────────
st.subheader("Daily Oil Price vs Housing Pressure Score")

recent_n = st.slider("Days of history", 30, 400, 90, step=30)
plot_df   = pressure[pressure["date"] >= pressure["date"].max() - pd.Timedelta(days=recent_n)].copy()
oil_plot  = oil_daily[oil_daily["date"] >= oil_daily["date"].max() - pd.Timedelta(days=recent_n)].copy()

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Oil area fill
fig.add_trace(go.Scatter(
    x=oil_plot["date"], y=oil_plot["oil_price_usd"],
    name="WTI Oil (daily)", line=dict(color="#f59e0b", width=2),
    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
), secondary_y=False)

# $100 reference
fig.add_hline(y=100, line_dash="dot", line_color="rgba(239,68,68,0.5)",
              annotation_text="$100", annotation_position="right", secondary_y=False)

# Pressure score
pos = plot_df[plot_df["housing_pressure_score"] >= 0]
neg = plot_df[plot_df["housing_pressure_score"] < 0]

fig.add_trace(go.Scatter(
    x=plot_df["date"], y=plot_df["housing_pressure_score"],
    name="Housing Pressure Score", line=dict(color="#60a5fa", width=2.5),
), secondary_y=True)

fig.add_trace(go.Scatter(
    x=pos["date"], y=pos["housing_pressure_score"],
    fill="tozeroy", fillcolor="rgba(34,197,94,0.15)",
    line=dict(width=0), showlegend=False, name="Positive pressure",
), secondary_y=True)

fig.add_trace(go.Scatter(
    x=neg["date"], y=neg["housing_pressure_score"],
    fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
    line=dict(width=0), showlegend=False, name="Negative pressure",
), secondary_y=True)

fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", secondary_y=True)

fig.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=False, zeroline=False)
fig.update_yaxes(title_text="WTI Oil (USD/barrel)", secondary_y=False,
                 showgrid=True, gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(title_text="Predicted MoM % change", secondary_y=True,
                 showgrid=False, zeroline=False)

st.plotly_chart(fig, use_container_width=True)


# ── Bottom row: CREB history + oil distribution ────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("CREB Benchmark Price History")
    creb_all = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    creb_tr2 = creb_all[creb_all["district"] == "Calgary"].copy()

    fig2 = go.Figure()
    colors = {"Detached": "#60a5fa", "Apartment": "#f59e0b",
              "Row": "#a78bfa", "Semi-Detached": "#34d399", "Total Residential": "#f87171"}
    for pt in ["Total Residential", "Detached", "Apartment", "Row", "Semi-Detached"]:
        sub = creb_tr2[creb_tr2["property_type"] == pt].sort_values("date")
        fig2.add_trace(go.Scatter(
            x=sub["date"], y=sub["benchmark_price"] / 1000,
            name=pt, line=dict(width=2.5 if pt == "Total Residential" else 1.5,
                               color=colors.get(pt, "#888"),
                               dash="solid" if pt == "Total Residential" else "dot"),
        ))
    fig2.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="$000s CAD", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                      tickprefix="$", ticksuffix="k")
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Oil Regime — Last 90 Days")
    oil_90 = oil_daily.tail(90).copy()
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=oil_90["oil_price_usd"], nbinsx=20,
        marker_color="#f59e0b", opacity=0.8, name="Oil price distribution",
    ))
    fig3.add_vline(x=oil_today, line_color="#ef4444", line_width=2,
                   annotation_text=f"Today ${oil_today:.0f}", annotation_position="top right")
    fig3.add_vline(x=oil_90["oil_price_usd"].mean(), line_dash="dot", line_color="#60a5fa",
                   annotation_text=f"90d avg ${oil_90['oil_price_usd'].mean():.0f}",
                   annotation_position="top left")
    fig3.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="WTI Oil (USD/barrel)", yaxis_title="Days",
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig3, use_container_width=True)


# ── Interpretation ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("How to read this")
i1, i2, i3 = st.columns(3)
i1.info("**Pressure Score > 0** — economic conditions favor rising benchmark prices next month. Oil moderate, rates stable.")
i2.warning("**Pressure Score ≈ 0** — neutral. Market momentum is the dominant driver, not macro signals.")
i3.error("**Pressure Score < 0** — economic shock regime. Oil spike or rate surge signals demand destruction, not prosperity. Current: oil at $100 flipped score negative Mar 10.")

st.caption("Data: CREB monthly stats PDFs · City of Calgary Open Data · Bank of Canada Valet API · WTI via yfinance · MiroFish geopolitical simulation")
