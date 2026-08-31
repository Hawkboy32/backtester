"""Charts crypto_btc_eth_5gbp_multiwindow.json's equity curves - VWAP Mean
Reversion (the one strategy that showed a real, multi-window-validated edge
on BTC) shown for both BTC and ETH side by side, default vs tuned params,
across all 3 non-overlapping windows, at the real $5 starting budget now in
CBAPI/KRKAPI. Same "Holotable" palette used throughout this project's
dashboard/mobile app/widget (hologram cyan #3dc7f0 + warm gold #e0a94a on
near-black #0b0e14) for visual consistency with everything else.

Run: python plot_crypto_5gbp_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

IN_JSON = Path(__file__).resolve().parent / "crypto_btc_eth_5gbp_multiwindow.json"
OUT_PNG = Path(__file__).resolve().parent / "crypto_5gbp_vwap_results.png"

COL_BG = "#0B0E14"
COL_ACCENT = "#3DC7F0"
COL_GOLD = "#E0A94A"
COL_DIM = "#5B7386"
COL_GRID = "#1C2733"
COL_TEXT = "#D6E4EE"

STRATEGY = "VWAP Mean Reversion"
TICKERS = ["X:BTCUSD", "X:ETHUSD"]
TICKER_LABELS = {"X:BTCUSD": "BTC", "X:ETHUSD": "ETH"}


def main() -> int:
    data = json.loads(IN_JSON.read_text())
    series = data["series"]
    starting_cash = data["starting_cash"]
    windows = data["windows"]

    fig = make_subplots(
        rows=2, cols=3,
        row_titles=["BTC", "ETH"],
        subplot_titles=[f"Window {i+1}<br><sub>{w[0]} → {w[1]}</sub>" for i, w in enumerate(windows)] * 1,
        shared_yaxes=False,
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
    )

    for row, ticker in enumerate(TICKERS, start=1):
        for col in range(1, 4):
            for variant, color, dash in (("default", COL_DIM, "dot"), ("tuned", COL_ACCENT if row == 1 else COL_GOLD, "solid")):
                key = f"{ticker}/{STRATEGY}/{variant}/window{col}"
                s = series.get(key)
                if not s:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=s["dates"], y=s["values"],
                        mode="lines",
                        name=f"{TICKER_LABELS[ticker]} {variant}",
                        line=dict(color=color, width=2.2, dash=dash),
                        showlegend=(col == 1),
                        legendgroup=f"{ticker}-{variant}",
                        hovertemplate="%{x}<br>$%{y:.3f}<extra>" + f"{TICKER_LABELS[ticker]} {variant}" + "</extra>",
                    ),
                    row=row, col=col,
                )
            fig.add_hline(
                y=starting_cash, line=dict(color=COL_DIM, width=1, dash="dash"),
                row=row, col=col, opacity=0.6,
            )

    fig.update_layout(
        title=dict(
            text=f"VWAP Mean Reversion — BTC vs ETH, default vs tuned params, ${starting_cash:.2f} start<br>"
                 f"<sub>3 non-overlapping ~8-month windows, liquidity-aware slippage, 100% sizing</sub>",
            font=dict(color=COL_TEXT, size=18), x=0.02,
        ),
        paper_bgcolor=COL_BG, plot_bgcolor=COL_BG,
        font=dict(color=COL_TEXT, family="Arial, sans-serif"),
        legend=dict(bgcolor=COL_BG, bordercolor=COL_GRID, borderwidth=1, font=dict(color=COL_TEXT)),
        width=1500, height=820,
        margin=dict(t=140, l=70, r=40, b=60),
    )
    fig.update_xaxes(gridcolor=COL_GRID, zerolinecolor=COL_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=COL_GRID, zerolinecolor=COL_GRID, tickprefix="$", tickfont=dict(size=10))
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(color=COL_TEXT, size=12)

    fig.write_image(str(OUT_PNG), scale=2)
    print(f"Saved: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
