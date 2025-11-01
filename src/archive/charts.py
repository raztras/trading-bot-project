import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from src.config import FILES_DIR, DAYS
from plotly.subplots import make_subplots  # added


def plot_chart(df: pd.DataFrame, hours: int = DAYS*24):
    dfx = df.copy()
    if not isinstance(dfx.index, pd.DatetimeIndex):
        dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce", utc=True)
        dfx = dfx.set_index("timestamp")
    dfx = dfx.sort_index()

    end = dfx.index.max()
    start = end - pd.Timedelta(hours=hours)
    dfx = dfx.loc[start:end]
    if dfx.empty:
        dfx = df.tail(60)  # fallback

    x_index = (
        dfx.index.tz_localize(None)
        if getattr(dfx.index, "tz", None) is not None
        else dfx.index
    )

    rsi_col = None
    if "rsi" in dfx.columns:
        rsi_col = "rsi"
    else:
        rsi_candidates = [c for c in dfx.columns if c.lower().startswith("rsi")]
        rsi_col = rsi_candidates[0] if rsi_candidates else None

    has_rsi = rsi_col is not None

    if has_rsi:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.72, 0.28],
        )
    else:
        fig = go.Figure()

    # Candles
    if has_rsi:
        fig.add_trace(
            go.Candlestick(
                x=x_index,
                open=dfx["open"],
                high=dfx["high"],
                low=dfx["low"],
                close=dfx["close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=x_index,
                open=dfx["open"],
                high=dfx["high"],
                low=dfx["low"],
                close=dfx["close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=False,
            )
        )

    for col, name, color in [
        ("ema_short", "EMA short", "#ff9800"),
        ("ema_long", "EMA long", "#6d4c41"),
        ("sma_fast", "SMA fast", "#009688"),
        ("sma_slow", "SMA slow", "#2196f3"),
    ]:
        if col in dfx.columns:
            trace = go.Scatter(
                x=x_index,
                y=dfx[col],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
            )
            if has_rsi:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

    if {"bb_upper", "bb_lower"}.issubset(dfx.columns):
        up = go.Scatter(
            x=x_index,
            y=dfx["bb_upper"],
            line=dict(color="rgba(33,150,243,0.6)", width=1),
            name="BB Upper",
        )
        lo = go.Scatter(
            x=x_index,
            y=dfx["bb_lower"],
            line=dict(color="rgba(33,150,243,0.6)", width=1),
            fill="tonexty",
            fillcolor="rgba(33,150,243,0.12)",
            name="BB Lower",
        )
        if has_rsi:
            fig.add_trace(up, row=1, col=1)
            fig.add_trace(lo, row=1, col=1)
        else:
            fig.add_trace(up)
            fig.add_trace(lo)
    if "bb_mid" in dfx.columns:
        mid = go.Scatter(
            x=x_index,
            y=dfx["bb_mid"],
            mode="lines",
            name="BB Mid",
            line=dict(color="#3f51b5", width=1),
        )
        if has_rsi:
            fig.add_trace(mid, row=1, col=1)
        else:
            fig.add_trace(mid)

    if has_rsi:
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=dfx[rsi_col],
                mode="lines",
                name=rsi_col.upper(),
                line=dict(color="#9c27b0", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=70,
            y1=70,
            line=dict(color="rgba(156,39,176,0.4)", width=1, dash="dash"),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=30,
            y1=30,
            line=dict(color="rgba(156,39,176,0.4)", width=1, dash="dash"),
        )

    fig.update_layout(
        title=f"Scalp view (last {hours}h)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out_dir = Path(FILES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_ts = dfx.index.max()
    ts_str = pd.Timestamp(end_ts).tz_localize(None).strftime("%Y%m%d_%H%M%S")
    base = f"scalp_view_{hours}h_{ts_str}"
    html_path = out_dir / f"{base}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    try:
        png_path = out_dir / f"{base}.png"
        fig.write_image(str(png_path), scale=2)
    except Exception:
        pass

    return str(html_path)
