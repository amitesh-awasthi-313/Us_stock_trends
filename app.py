# app.py
import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Trends Viewer", layout="wide")

# ---------- Config ----------
DEFAULT_DATA_DIR = "companies_csv_v2"  # set your folder once here
VALID_PRICE_COLS = ["Adj Close", "Close"]
BASE_FIELDS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def list_tickers(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    tickers = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return tickers, files

@st.cache_data(show_spinner=False)
def load_ticker_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure expected columns exist
    if "Date" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Force presence of expected numeric columns (fill with NaN if missing)
    for c in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def normalize_series(s: pd.Series, mode: str) -> pd.Series:
    if mode == "None":
        return s
    s = s.astype(float)
    if mode == "Z-score":
        mu, sigma = s.mean(), s.std(ddof=0)
        return (s - mu) / (sigma if sigma and not np.isnan(sigma) else 1)
    if mode == "Min-Max":
        mn, mx = s.min(), s.max()
        rng = mx - mn
        return (s - mn) / (rng if rng and not np.isnan(rng) else 1)
    if mode == "% Change":
        return s.pct_change().fillna(0)
    return s

def apply_theme(dark: bool):
    """Light/Dark theme via CSS + Plotly template name."""
    if dark:
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: #f0f2f6; }
            .stMarkdown, .stRadio, .stSelectbox, .stMultiSelect, .stDateInput, .stCaption, .stButton, .stText { color: #f0f2f6 !important; }
            </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #ffffff; color: #1f1f1f; }
            </style>
        """, unsafe_allow_html=True)
        return "plotly"

def finalize_interactions(fig, template, height=600):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        template=template,
        hovermode="x unified",
        hoverdistance=50,
        transition_duration=500
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y}")
    return fig

def build_overlay_chart(df_map, price_col, norm, date_from, date_to, show_volume, template):
    fig = make_subplots(specs=[[{"secondary_y": show_volume}]])
    for ticker, df in df_map.items():
        mask = (df["Date"] >= date_from) & (df["Date"] <= date_to)
        d = df.loc[mask, ["Date", price_col, "Volume"]].copy()
        d[price_col] = normalize_series(d[price_col], norm)
        fig.add_trace(
            go.Scatter(x=d["Date"], y=d[price_col], name=ticker, mode="lines"),
            secondary_y=False
        )
    if show_volume:
        for ticker, df in df_map.items():
            mask = (df["Date"] >= date_from) & (df["Date"] <= date_to)
            d = df.loc[mask, ["Date", "Volume"]].copy()
            fig.add_trace(
                go.Bar(x=d["Date"], y=d["Volume"], name=f"{ticker} Vol", opacity=0.25, showlegend=False),
                secondary_y=True
            )
        fig.update_yaxes(title_text="Volume", secondary_y=True)

    fig.update_yaxes(title_text=f"{price_col} ({norm})", secondary_y=False)
    return finalize_interactions(fig, template, height=600)

def build_small_multiples(df_map, price_col, norm, date_from, date_to, template, cols=3):
    tickers = list(df_map.keys())
    rows = int(np.ceil(len(tickers) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tickers, shared_xaxes=True)
    r = c = 1
    for ticker in tickers:
        df = df_map[ticker]
        mask = (df["Date"] >= date_from) & (df["Date"] <= date_to)
        d = df.loc[mask, ["Date", price_col]].copy()
        d[price_col] = normalize_series(d[price_col], norm)
        fig.add_trace(
            go.Scatter(x=d["Date"], y=d[price_col], name=ticker, mode="lines", showlegend=False),
            row=r, col=c
        )
        c += 1
        if c > cols:
            c = 1
            r += 1
    return finalize_interactions(fig, template, height=max(600, 220 * rows))

# ---------- UI ----------
st.title("US Stocks – Trend Viewer (Phase 1)")
st.caption("Browse weekly trends; normalize; filter by date; compare visually.")

with st.sidebar:
    # Dark/Light toggle: default = True (dark)
    dark_mode = st.toggle("Dark mode", value=True)
    template = apply_theme(dark_mode)

    st.header("Data")
    # Hide folder input; use constant dir
    data_dir = DEFAULT_DATA_DIR
    tickers, files = list_tickers(data_dir)
    if not tickers:
        st.error(f"No CSV files found in '{data_dir}'. Put your per-company CSVs there.")
        st.stop()

    st.write(f"Found {len(tickers)} companies.")

    # Default: show only 1 company initially
    select_all = st.checkbox("Select all companies", value=False)
    default_selection = [tickers[0]] if not select_all else tickers
    selected = st.multiselect("Companies", options=tickers, default=default_selection)
    if select_all and len(selected) != len(tickers):
        selected = tickers

    price_col = st.selectbox("Price column", VALID_PRICE_COLS, index=0)
    norm = st.selectbox("Normalization", ["None", "Z-score", "Min-Max", "% Change"], index=0)
    layout_mode = st.radio("Layout", ["Overlay (one chart)", "Small multiples"], index=0)
    show_volume = st.checkbox("Show volume (overlay mode only)", value=False)

# Load selected data
df_map = {}
for ticker in selected:
    path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(path):
        df_map[ticker] = load_ticker_csv(path)

if not df_map:
    st.warning("No data loaded for selected companies.")
    st.stop()

# Global date range from loaded data
min_date = min(df["Date"].min() for df in df_map.values())
max_date = max(df["Date"].max() for df in df_map.values())
col1, col2 = st.columns(2)
with col1:
    d1 = st.date_input("Start date", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
with col2:
    d2 = st.date_input("End date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
date_from, date_to = pd.to_datetime(d1), pd.to_datetime(d2)

# Charts
if layout_mode.startswith("Overlay"):
    fig = build_overlay_chart(df_map, price_col, norm, date_from, date_to, show_volume, template)
else:
    fig = build_small_multiples(df_map, price_col, norm, date_from, date_to, template)

st.plotly_chart(fig, use_container_width=True)

# Table + Download
st.subheader("Data preview")
combined = []
for t, df in df_map.items():
    m = (df["Date"] >= date_from) & (df["Date"] <= date_to)
    d = df.loc[m, ["Date", price_col, "High", "Low", "Open", "Volume"]].copy()
    d.insert(0, "Ticker", t)
    if norm != "None":
        d[price_col] = normalize_series(d[price_col], norm)
    combined.append(d)
out_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

st.dataframe(out_df.head(500), use_container_width=True)
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="filtered_trends.csv", mime="text/csv")
