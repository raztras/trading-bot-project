import numpy as np
import pandas as pd


def simulate_trades(
    df: pd.DataFrame,
    start_cash: float = 100_000.0,
    buy_fee: float = 0.001,  # 0.1%
    sell_fee: float = 0.001,  # 0.1%
    target_pct: float = 0.01,  # 0.5% TP
    stop_loss_pct: float = 0.0075,  # 0.5% SL (tune this)
    price_col: str = "close",
    buy_col: str = "buy",
    sell_col: str = "sell",
    use_high_for_target: bool = True,
    use_low_for_stop: bool = True,
    exit_on_sell_signal: bool = True,
    cooldown_bars: int = 2,  # wait after exit before next entry
    require_profit_on_sell_signal: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Spot simulation with TP/SL:
      - Enter long on buy==True using all cash (buy fee applied).
      - Exit on intrabar stop-loss (uses 'low'), or take-profit (uses 'high'),
        or optionally on sell==True (sell fee applied).
      - cooldown_bars prevents immediate re-entry after exit.
    """
    dfx = df.copy()

    required = {price_col, buy_col, sell_col, "high", "low"}
    missing = required - set(dfx.columns)
    if missing:
        raise ValueError(f"simulate_trades: missing columns: {missing}")

    cash = float(start_cash)
    qty = 0.0
    in_pos = False
    entry_px = None
    entry_i = None
    cooldown = 0

    position = []
    entry_price_track = []
    equity = []
    cash_track = []
    qty_track = []
    simulation_track = []

    trades: list[dict] = []
    realized_cash_value = float(start_cash)

    for i, row in dfx.iterrows():
        price = float(row[price_col])
        high = float(row["high"])
        low = float(row["low"])

        # decrement cooldown
        if cooldown > 0:
            cooldown -= 1

        # Enter on BUY signal if flat and not cooling down
        if not in_pos and cooldown == 0 and bool(row[buy_col]):
            entry_px = price
            entry_i = i
            spendable = cash * (1.0 - buy_fee)  # fee taken from cash
            qty = spendable / entry_px if entry_px > 0 else 0.0
            cash = 0.0
            in_pos = True

        if in_pos:
            tp_price = entry_px * (1.0 + target_pct)
            sl_price = entry_px * (1.0 - stop_loss_pct)

            # Conservative ordering: hit stop first if low breaches, else TP if high breaches
            if use_low_for_stop and low <= sl_price:
                exit_px = sl_price
                proceeds = qty * exit_px * (1.0 - sell_fee)
                pnl = proceeds - (qty * entry_px)
                cash += proceeds
                trades.append(
                    dict(
                        entry_index=entry_i,
                        entry_price=entry_px,
                        exit_index=i,
                        exit_price=exit_px,
                        reason="SL",
                        gross_ret=(exit_px / entry_px) - 1.0,
                        net_ret=(exit_px / entry_px) * (1 - buy_fee) * (1 - sell_fee)
                        - 1.0,
                        qty=qty,
                        cash_out=proceeds,
                        realized_pnl=pnl,
                    )
                )
                qty = 0.0
                in_pos = False
                entry_px = None
                entry_i = None
                cooldown = cooldown_bars

            elif use_high_for_target and high >= tp_price:
                exit_px = tp_price
                proceeds = qty * exit_px * (1.0 - sell_fee)
                pnl = proceeds - (qty * entry_px)
                cash += proceeds
                trades.append(
                    dict(
                        entry_index=entry_i,
                        entry_price=entry_px,
                        exit_index=i,
                        exit_price=exit_px,
                        reason="TP",
                        gross_ret=(exit_px / entry_px) - 1.0,
                        net_ret=(exit_px / entry_px) * (1 - buy_fee) * (1 - sell_fee)
                        - 1.0,
                        qty=qty,
                        cash_out=proceeds,
                        realized_pnl=pnl,
                    )
                )
                qty = 0.0
                in_pos = False
                entry_px = None
                entry_i = None
                cooldown = cooldown_bars

            elif exit_on_sell_signal and bool(row[sell_col]):
                exit_px = price
                # Net return including both entry (buy_fee) and exit (sell_fee)
                net_ret_now = (exit_px / entry_px) * (1 - buy_fee) * (1 - sell_fee) - 1.0
                if (not require_profit_on_sell_signal) or (net_ret_now >= 0.0):
                    proceeds = qty * exit_px * (1.0 - sell_fee)
                    pnl = proceeds - (qty * entry_px)
                    cash += proceeds
                    trades.append(
                        dict(
                            entry_index=entry_i,
                            entry_price=entry_px,
                            exit_index=i,
                            exit_price=exit_px,
                            reason="SELL_SIGNAL",
                            gross_ret=(exit_px / entry_px) - 1.0,
                            net_ret=net_ret_now,
                            qty=qty,
                            cash_out=proceeds,
                            realized_pnl=pnl,
                        )
                    )
                    qty = 0.0
                    in_pos = False
                    entry_px = None
                    entry_i = None
                    cooldown = cooldown_bars

        # Tracking
        position.append(1 if in_pos else 0)
        entry_price_track.append(entry_px if in_pos else np.nan)
        cash_track.append(cash)
        qty_track.append(qty)
        equity.append(cash + qty * price)

        if not in_pos:
            realized_cash_value = cash
        simulation_track.append(realized_cash_value)

    # Annotate
    dfx["position"] = position
    dfx["entry_price"] = entry_price_track
    dfx["cash"] = cash_track
    dfx["qty"] = qty_track
    dfx["equity"] = equity
    dfx["simulation"] = simulation_track

    trades_df = pd.DataFrame(trades)
    summary = dict(
        start_cash=start_cash,
        end_cash=cash,
        end_equity=dfx["equity"].iloc[-1],
        n_trades=len(trades_df),
        win_rate=float((trades_df["net_ret"] > 0).mean())
        if not trades_df.empty
        else 0.0,
        avg_net_ret=float(trades_df["net_ret"].mean()) if not trades_df.empty else 0.0,
        total_net_ret=float((dfx["equity"].iloc[-1] / start_cash) - 1.0),
        params=dict(
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            target_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
        ),
    )

    return dfx, trades_df, summary
