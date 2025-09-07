"""
ChatGPT Micro-Cap Experiment — Borsa İstanbul (BIST) Edition
-----------------------------------------------------------
A self-contained trading utility adapted for the Turkish stock market.

What this provides (parity with original, adapted to TR):
- Centralized market data accessor (Yahoo first; graceful fallback stubs)
- Weekend/holiday-safe "as-of" date handling
- TRY-denominated portfolio (CSV-based state), trade log, equity history
- Benchmarks: XU100.IS (BIST 100), USDTRY=X
- Daily prices snapshot CSVs under ./daily_results/
- Trailing stop-loss automation (configurable); per-position hard stops supported
- Console "Daily Results" report you can paste into ChatGPT
- Simple order parser to APPLY ChatGPT's BUY/SELL/HOLD/STOP instructions
- ASOF_DATE override for backtesting/testing reproducibility

NOTE
- Yahoo Finance supports BIST via ".IS" tickers (e.g., THYAO.IS). This script
  uses yfinance and handles empty frames/temporary errors gracefully.
- "Stooq" fallback commonly used in the US version may not carry BIST coverage.
  We keep a stub fallback that simply returns None and logs a warning so the
  behavior is uniform and extensible if you later add another source.

Python: 3.11+
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------------- Global Config -----------------------------

# Optional global override: treat this date as "today" (YYYY-MM-DD)
ASOF_DATE: Optional[pd.Timestamp] = None

# Folders / files
DATA_DIR = Path("daily_results")
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_CSV = Path("portfolio.csv")
TRADES_CSV = Path("trades.csv")
EQUITY_CSV = Path("equity.csv")
SETTINGS_JSON = Path("settings.json")

# Defaults if settings.json not present
DEFAULT_SETTINGS = {
    "base_currency": "TRY",
    "starting_cash": 1000.0,        # Adjust as you like
    "max_positions": 6,               # Keep it small; liquidity matters
    "default_trailing_stop_pct": 0.12,  # 12% trailing stop
    "commission_per_trade": 0.0,      # Simulated
    "slippage_bps": 0,                # Simulated; 10 bps = 0.001
    "report_benchmarks": True,
    "universe": "bist_curated",
}

# Core Turkish universe (liquid BIST100 + a few small/speculative names)
BIST_TICKERS = [
    # Banks & Finance
    "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "ISCTR.IS", "HALKB.IS", "VAKBN.IS",
    # Industrials / Materials / Energy
    "EREGL.IS", "KRDMD.IS", "TUPRS.IS", "PETKM.IS", "ALARK.IS",
    # Export & Manufacturing
    "VESTL.IS", "ARCLK.IS", "SASA.IS", "KORDS.IS",
    # Defense & Tech
    "ASELS.IS", "OTKAR.IS", "KAREL.IS",
    # Consumer & Services
    "BIMAS.IS", "MGROS.IS", "TAVHL.IS", "THYAO.IS",

    "PGSUS.IS", "IZMDC.IS"   # NEW core tickers
]

# Satellite small caps (speculative "moonshot" bucket; use tiny sizing)
BIST_SATELLITE = ["KONTR.IS", "GESAN.IS", "PENTA.IS", "QUAGR.IS", "KLSER.IS",
                  "BFREN.IS", "KRTEK.IS", "TRHOL.IS", "BARMA.IS", "BLCYT.IS",
                    "EFORC.IS", "TGSAS.IS", "PKENT.IS", "YYAPI.IS", "OBAMS.IS",
                    "PCILT.IS", "DSTKF.IS"   # NEW satellite tickers
                  ]

# Benchmarks
BENCHMARKS = ["XU100.IS", "USDTRY=X"]


def classify_universe(ticker: str) -> str:
    if ticker in BIST_TICKERS:
        return "CORE"
    elif ticker in BIST_SATELLITE:
        return "SATELLITE"
    elif ticker in BENCHMARKS:
        return "BENCHMARK"
    return "UNKNOWN"

# --------------------------- Data Structures -----------------------------

@dataclass
class Position:
    ticker: str
    shares: float
    buy_price: float
    stop_loss: float         # absolute price floor (optional; 0 = off)
    trailing_stop_pct: float # percent as decimal (e.g., 0.12)
    peak_price: float        # highest close since entry for trailing calc
    notes: str = ""


@dataclass
class Trade:
    date: str
    ticker: str
    side: str           # BUY / SELL / AUTO_SELL / STOP_ADJUST
    qty: float
    price: float
    commission: float
    slippage_bps: int
    reason: str = ""    # free text (e.g., "stop-loss", "chatgpt plan", etc.)
    universe: str = ""  # "CORE", "SATELLITE", or "BENCHMARK"

# ------------------------------- Settings --------------------------------

def load_settings() -> dict:
    if SETTINGS_JSON.exists():
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                s = json.load(f)
                return {**DEFAULT_SETTINGS, **s}
        except Exception as e:
            logging.warning(f"Failed reading settings.json; using defaults. {e}")
    return DEFAULT_SETTINGS.copy()


# ---------------------------- Utility / Dates ----------------------------

def set_asof(date: str | datetime | pd.Timestamp | None) -> None:
    """Set global 'as of' date so the script treats that day as 'today'."""
    global ASOF_DATE
    if date is None:
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()


def today_local() -> pd.Timestamp:
    return (ASOF_DATE or pd.Timestamp.today()).normalize()


def last_trading_day(ts: pd.Timestamp) -> pd.Timestamp:
    # Weekend handling (Turkey exchanges: Mon-Fri; holidays not enumerated here)
    wd = ts.weekday()  # Mon=0 ... Sun=6
    if wd == 5:   # Saturday
        return ts - pd.Timedelta(days=1)
    if wd == 6:   # Sunday
        return ts - pd.Timedelta(days=2)
    return ts


# --------------------------- Market Data Layer ---------------------------

class MarketData:
    """Centralizes price fetching; normalizes output columns."""

    def __init__(self):
        # yfinance: auto_adjust defaults may change; we control explicitly
        self._yahoo_auto_adjust = False

    def history(
        self,
        ticker: str,
        period: str = "5d",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch recent history. Try Yahoo; if empty/broken, fall back stub."""
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=self._yahoo_auto_adjust,
                progress=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                    }
                )
                return df
            logging.warning(f"Yahoo returned empty for {ticker}. Trying fallback...")
            # --- Fallback stub (placeholder) ---
            # In the US version you might use Stooq. BIST coverage is spotty there.
            # We keep a uniform signature so you can drop-in a real fallback later.
            return self._fallback_stub(ticker, period, interval)
        except Exception as e:
            logging.warning(f"Yahoo error for {ticker}: {e}. Trying fallback...")
            return self._fallback_stub(ticker, period, interval)

    @staticmethod
    def _fallback_stub(ticker: str, period: str, interval: str) -> pd.DataFrame:
        # No second source configured for BIST; return empty with correct columns.
        cols = ["open", "high", "low", "close", "adj_close", "volume"]
        return pd.DataFrame(columns=cols)


# --------------------------- Portfolio I/O --------------------------------

def load_portfolio() -> Dict[str, Position]:
    if not PORTFOLIO_CSV.exists():
        # Initialize empty portfolio file
        pd.DataFrame(
            columns=[
                "ticker", "shares", "buy_price", "stop_loss",
                "trailing_stop_pct", "peak_price", "notes"
            ]
        ).to_csv(PORTFOLIO_CSV, index=False)
        return {}

    df = pd.read_csv(PORTFOLIO_CSV)
    positions: Dict[str, Position] = {}
    for _, r in df.iterrows():
        positions[r["ticker"]] = Position(
            ticker=r["ticker"],
            shares=float(r["shares"]),
            buy_price=float(r["buy_price"]),
            stop_loss=float(r["stop_loss"]),
            trailing_stop_pct=float(r["trailing_stop_pct"]),
            peak_price=float(r["peak_price"]),
            notes=str(r.get("notes", "")),
        )
    return positions


def save_portfolio(positions: Dict[str, Position]) -> None:
    rows = [asdict(p) for p in positions.values()]
    df = pd.DataFrame(rows)
    df.to_csv(PORTFOLIO_CSV, index=False)


def load_cash(settings: dict) -> float:
    # Cash is stored in equity.csv last row or inferred. Simpler: keep a json state.
    cash_file = Path("cash.txt")
    if not cash_file.exists():
        cash_file.write_text(str(settings["starting_cash"]), encoding="utf-8")
        return float(settings["starting_cash"])
    try:
        return float(cash_file.read_text(encoding="utf-8").strip())
    except Exception:
        return float(settings["starting_cash"])


def save_cash(cash: float) -> None:
    Path("cash.txt").write_text(f"{cash:.2f}", encoding="utf-8")


# ------------------------- Reporting / Snapshots --------------------------

def snapshot_prices(
    md: MarketData,
    tickers: List[str],
    asof: pd.Timestamp
) -> pd.DataFrame:
    """Get last close, % change (d/d), and volume, tolerant of missing/misaligned data."""
    rows = []
    for t in tickers:
        h = md.history(t, period="6d", interval="1d")

        
        # Flatten MultiIndex columns if present
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)

        if h.empty:
            logging.warning(f"No data for {t}")
            rows.append({"ticker": t, "close": np.nan, "pct_chg": np.nan, "volume": np.nan})
            continue

        # --- Handle Yahoo column weirdness ---
        if "close" not in h.columns:
            if "adj_close" in h.columns:
                h["close"] = h["adj_close"]
            else:
                logging.warning(f"No usable close/adj_close column for {t}")
                rows.append({"ticker": t, "close": np.nan, "pct_chg": np.nan, "volume": np.nan})
                continue

        h = h.dropna(subset=["close"])
        if h.empty:
            rows.append({"ticker": t, "close": np.nan, "pct_chg": np.nan, "volume": np.nan})
            continue

        last = h.iloc[-1]
        prev = h.iloc[-2] if len(h) > 1 else None

        close = float(last["close"])
        vol = float(last.get("volume", np.nan)) if "volume" in last else np.nan
        pct = float(((close / float(prev["close"])) - 1.0) * 100.0) if prev is not None else 0.0

        rows.append({"ticker": t, "close": close, "pct_chg": pct, "volume": vol})

    df = pd.DataFrame(rows)
    out_file = DATA_DIR / f"{asof.date()}_prices.csv"
    df.to_csv(out_file, index=False)
    return df



def print_daily_report(asof: pd.Timestamp, prices: pd.DataFrame, portfolio_value_block: str) -> None:
    date_str = asof.strftime("%Y-%m-%d")
    sep = "=" * 64
    print(sep)
    print(f"Daily Results — {date_str}")
    print(sep)

    def print_table(df: pd.DataFrame, title: str):
        print(f"\n[ {title} ]")
        print(f"{'Ticker':<16}{'Close':>10}{'% Chg':>10}{'Volume':>14}")
        print("-" * 49)
        for _, r in df.sort_values("ticker").iterrows():
            close = "—" if pd.isna(r["close"]) else f"{r['close']:.2f}"
            pct = "—" if pd.isna(r["pct_chg"]) else f"{r['pct_chg']:+.2f}%"
            vol = "—" if pd.isna(r["volume"]) else f"{int(r['volume']):,}"
            print(f"{r['ticker']:<16}{close:>10}{pct:>10}{vol:>14}")

    # Split core vs satellite
    core_df = prices[prices["ticker"].isin(BIST_TICKERS)]
    sat_df = prices[prices["ticker"].isin(BIST_SATELLITE)]
    bench_df = prices[prices["ticker"].isin(BENCHMARKS)]

    if not core_df.empty:
        print_table(core_df, "Core Universe (Stable/Liquid)")

    if not sat_df.empty:
        print_table(sat_df, "Satellite Universe (Speculative)")

    if not bench_df.empty:
        print_table(bench_df, "Benchmarks")

    print("\n[ Portfolio Snapshot ]")
    print(portfolio_value_block)

    print("\n[ Request ]")
    print("Please give me buy/sell/hold recommendations based on this report.")

# --------------------------- Equity & Benchmarks --------------------------

def compute_equity_block(
    md: MarketData,
    positions: Dict[str, Position],
    cash: float,
    asof: pd.Timestamp,
    report_benchmarks: bool,
) -> Tuple[str, float]:
    """Returns printable block + latest total equity, with Core vs Satellite splits."""
    rows = []
    total_pos_val = 0.0
    core_val = 0.0
    sat_val = 0.0

    for p in positions.values():
        h = md.history(p.ticker, period="6d", interval="1d")

        # Fallback handling for close price
        if h.empty:
            px = np.nan
        elif "close" in h.columns:
            px = float(h["close"].iloc[-1])
        elif "adj_close" in h.columns:
            px = float(h["adj_close"].iloc[-1])
        else:
            px = np.nan

        val = 0.0 if np.isnan(px) else px * p.shares
        total_pos_val += val

        # Track by universe
        if p.ticker in BIST_TICKERS:
            core_val += val
        elif p.ticker in BIST_SATELLITE:
            sat_val += val

        rows.append([p.ticker, p.shares, p.stop_loss, p.buy_price, val])

    df = pd.DataFrame(rows, columns=["ticker", "shares", "stop_loss", "buy_price", "position_value"])
    df["cost_basis"] = df["shares"] * df["buy_price"]
    printable = df[["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]].to_string(index=False)

    total_equity = cash + total_pos_val
    equity_row = {
        "date": asof.strftime("%Y-%m-%d"),
        "equity_total": round(total_equity, 2),
        "equity_core": round(core_val, 2),
        "equity_satellite": round(sat_val, 2),
        "cash": round(cash, 2),
    }

    # Save equity.csv with new schema
    if EQUITY_CSV.exists():
        eq_df = pd.read_csv(EQUITY_CSV)
        eq_df = pd.concat([eq_df, pd.DataFrame([equity_row])], ignore_index=True)
    else:
        eq_df = pd.DataFrame([equity_row])
    eq_df.to_csv(EQUITY_CSV, index=False)

    # Benchmarks quick line
    if report_benchmarks:
        bench_lines = []
        for b in BENCHMARKS:
            hb = md.history(b, period="6d", interval="1d")
            if hb.empty:
                bench_lines.append(f"{b}: n/a")
                continue
            if "close" in hb.columns:
                last = float(hb["close"].iloc[-1])
                prev = float(hb["close"].iloc[-2]) if len(hb) > 1 else last
            elif "adj_close" in hb.columns:
                last = float(hb["adj_close"].iloc[-1])
                prev = float(hb["adj_close"].iloc[-2]) if len(hb) > 1 else last
            else:
                bench_lines.append(f"{b}: n/a")
                continue
            chg = ((last / prev) - 1.0) * 100.0
            bench_lines.append(f"{b}: {last:.2f} ({chg:+.2f}%)")
        printable += f"\nBenchmarks -> " + " | ".join(bench_lines)

    return printable, total_equity



# --------------------------- Stop-Loss Engine -----------------------------

def apply_trailing_stops(
    md: MarketData,
    positions: Dict[str, Position],
    asof: pd.Timestamp,
    commission_per_trade: float,
    slippage_bps: int,
) -> Tuple[Dict[str, Position], List[Trade], float]:
    """Update peaks; auto-sell if violated. Returns (updated_positions, trades, realized_pnl_delta)"""
    trades: List[Trade] = []
    pnl_delta = 0.0
    updated = positions.copy()

    for t, p in list(updated.items()):
        h = md.history(t, period="30d", interval="1d")
        if h.empty or "close" not in h:
            logging.warning(f"No price for {t}; skipping stop eval.")
            continue
        last_close = float(h["close"].iloc[-1])

        # Update peak
        p.peak_price = float(max(p.peak_price, last_close)) if p.peak_price > 0 else last_close

        # Compute effective trailing stop
        trailing_floor = p.peak_price * (1.0 - p.trailing_stop_pct)
        effective_floor = trailing_floor
        if p.stop_loss and p.stop_loss > 0:
            effective_floor = max(effective_floor, p.stop_loss)  # choose the higher floor

        # Trigger?
        if last_close <= effective_floor and p.shares > 0:
            # Simulate market execution ~ close price +/- slippage
            # (slippage positive for buys; negative for sells in bps)
            slip_mult = 1.0 - (slippage_bps / 10000.0)
            exec_px = last_close * slip_mult
            qty = p.shares

            proceeds = exec_px * qty - commission_per_trade
            cost = p.buy_price * qty
            pnl = proceeds - cost
            pnl_delta += pnl

            trades.append(
                Trade(
                    date=asof.strftime("%Y-%m-%d"),
                    ticker=t,
                    side="AUTO_SELL",
                    qty=float(qty),
                    price=round(exec_px, 4),
                    commission=commission_per_trade,
                    slippage_bps=slippage_bps,
                    reason=f"Stop triggered (floor={effective_floor:.2f}, peak={p.peak_price:.2f})",
                    universe=classify_universe(t),
                )
            )
            # Remove / zero the position
            del updated[t]

    return updated, trades, pnl_delta


# --------------------------- Orders & Execution ---------------------------

def parse_orders(text: str) -> List[Tuple[str, List[str]]]:
    """
    Accept commands like:
      BUY THYAO.IS 25 @ MKT
      SELL AKBNK.IS ALL @ LMT 37.50
      HOLD GARAN.IS
      STOP THYAO.IS 12%           (sets trailing stop to 12%)
      HARDSTOP THYAO.IS 78.50     (absolute floor)
    Returns list of (cmd, tokens)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed: List[Tuple[str, List[str]]] = []
    for ln in lines:
        toks = ln.replace(",", " ").split()
        cmd = toks[0].upper()
        parsed.append((cmd, toks[1:]))
    return parsed


def apply_orders(
    md: MarketData,
    positions: Dict[str, Position],
    cash: float,
    settings: dict,
    orders_txt: str,
    asof: pd.Timestamp,
) -> Tuple[Dict[str, Position], float, List[Trade]]:
    parsed = parse_orders(orders_txt)
    trades: List[Trade] = []
    commission = float(settings["commission_per_trade"])
    bps = int(settings["slippage_bps"])

    # Helper to exec at last close (approx). For MOO you can switch to "Open" with intraday provider.
    def last_price(t: str) -> float:
        h = md.history(t, period="6d", interval="1d")
        if h.empty or "close" not in h:
            raise ValueError(f"No price for {t}")
        return float(h["close"].iloc[-1])

    for cmd, toks in parsed:
        if cmd == "HOLD":
            # No action; optional: ensure position exists
            continue

        if cmd in ("BUY", "SELL"):
            if len(toks) < 1:
                continue
            ticker = toks[0].upper()
            qty: Optional[float] = None
            # Find "ALL"
            if len(toks) >= 2 and toks[1].upper() == "ALL":
                qty = -1.0  # sentinel
            elif len(toks) >= 2:
                try:
                    qty = float(toks[1])
                except Exception:
                    qty = None

            # Price handling - here we just use market-at-close approximation.
            px = last_price(ticker)

            # Slippage: buys pay more, sells receive less
            if cmd == "BUY":
                px *= (1.0 + bps / 10000.0)
            else:
                px *= (1.0 - bps / 10000.0)

            if cmd == "BUY":
                if qty is None:
                    raise ValueError(f"BUY {ticker} missing qty")
                if qty <= 0:
                    raise ValueError(f"BUY qty must be positive for {ticker}")

                # Cash check
                cost = px * qty + commission
                if cost > cash + 1e-6:
                    raise ValueError(f"Insufficient cash for BUY {ticker} x {qty}")

                if ticker in positions:
                    # Add to existing (average price)
                    p = positions[ticker]
                    new_shares = p.shares + qty
                    new_cost = p.buy_price * p.shares + px * qty
                    p.buy_price = new_cost / new_shares
                    p.shares = new_shares
                    p.peak_price = max(p.peak_price, px)
                else:
                    positions[ticker] = Position(
                        ticker=ticker,
                        shares=qty,
                        buy_price=px,
                        stop_loss=0.0,
                        trailing_stop_pct=float(settings["default_trailing_stop_pct"]),
                        peak_price=px,
                        notes="",
                    )
                cash -= cost
                trades.append(Trade(asof.strftime("%Y-%m-%d"), ticker, "BUY", qty, round(px, 4), commission, bps, "chatgpt plan",classify_universe(ticker)))

            elif cmd == "SELL":
                if ticker not in positions:
                    # Nothing to sell
                    continue
                p = positions[ticker]
                if qty is None or qty < 0 or qty > p.shares or toks[1].upper() == "ALL":
                    qty = p.shares

                proceeds = px * qty - commission
                p.shares -= qty
                if p.shares <= 1e-9:
                    del positions[ticker]
                cash += proceeds
                trades.append(Trade(asof.strftime("%Y-%m-%d"), ticker, "SELL", qty, round(px, 4), commission, bps, "chatgpt plan", classify_universe(ticker)))

        elif cmd == "STOP":
            # Set trailing stop percent, e.g., STOP THYAO.IS 12%
            if len(toks) < 2:
                continue
            ticker = toks[0].upper()
            pct_str = toks[1].replace("%", "")
            try:
                pct = float(pct_str) / 100.0
            except Exception:
                continue
            if ticker in positions:
                positions[ticker].trailing_stop_pct = pct
                trades.append(Trade(asof.strftime("%Y-%m-%d"), ticker, "STOP_ADJUST", 0, 0.0, 0.0, 0, f"trailing={pct:.4f}", classify_universe(ticker)))

        elif cmd == "HARDSTOP":
            # Set absolute price floor, e.g., HARDSTOP THYAO.IS 78.50
            if len(toks) < 2:
                continue
            ticker = toks[0].upper()
            try:
                floor = float(toks[1])
            except Exception:
                continue
            if ticker in positions:
                positions[ticker].stop_loss = floor
                trades.append(Trade(asof.strftime("%Y-%m-%d"), ticker, "STOP_ADJUST", 0, 0.0, 0.0, 0, f"hard={floor:.2f}", classify_universe(ticker)))

        else:
            logging.warning(f"Unknown command: {cmd} ({' '.join(toks)})")

    return positions, cash, trades


def append_trades(trades: List[Trade]) -> None:
    if not trades:
        return
    rows = [asdict(t) for t in trades]
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(TRADES_CSV, index=False)


# ------------------------------- Main Flow --------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BIST Trading Script (ChatGPT Micro-Cap Experiment — TR)")
    parser.add_argument("--mode", choices=["report", "apply", "stops-only"], default="report",
                        help="report: print daily report; apply: apply orders from --orders; stops-only: only evaluate stops")
    parser.add_argument("--orders", type=str, help="Path to a text file with orders (BUY/SELL/HOLD/STOP/HARDSTOP)")
    parser.add_argument("--asof", type=str, help="YYYY-MM-DD override date")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Handle ASOF
    if args.asof:
        set_asof(args.asof)

    settings = load_settings()
    md = MarketData()
    positions = load_portfolio()
    cash = load_cash(settings)

    asof = last_trading_day(today_local())

    # 1) Apply trailing stops automatically (unless just reporting without changes)
    if args.mode in ("report", "stops-only", "apply"):
        positions, stop_trades, pnl_delta = apply_trailing_stops(
            md, positions, asof,
            commission_per_trade=float(settings["commission_per_trade"]),
            slippage_bps=int(settings["slippage_bps"]),
        )
        if stop_trades:
            append_trades(stop_trades)
            # Increase cash by proceeds from AUTO_SELL trades
            # We recompute cash delta from trades to be exact:
            cash_delta = 0.0
            for t in stop_trades:
                if t.side == "AUTO_SELL":
                    cash_delta += t.price * t.qty - t.commission
            cash += cash_delta
            save_cash(cash)
            save_portfolio(positions)

        if args.mode == "stops-only":
            # Still snapshot & show quick results
            tickers_for_report = sorted(set(BIST_TICKERS + BIST_SATELLITE + (BENCHMARKS if settings["report_benchmarks"] else [])))
            prices = snapshot_prices(md, tickers_for_report, asof)
            port_block, _ = compute_equity_block(md, positions, cash, asof, settings["report_benchmarks"])
            print_daily_report(asof, prices, port_block)
            return 0

    # 2) If apply mode, read orders file and execute
    if args.mode == "apply":
        if not args.orders:
            logging.error("--orders is required in apply mode")
            return 2
        txt = Path(args.orders).read_text(encoding="utf-8")
        positions, cash, plan_trades = apply_orders(md, positions, cash, settings, txt, asof)
        if plan_trades:
            append_trades(plan_trades)
        save_portfolio(positions)
        save_cash(cash)

    # 3) Always produce a fresh report snapshot for ChatGPT
    tickers_for_report = sorted(set(BIST_TICKERS + BIST_SATELLITE + (BENCHMARKS if settings["report_benchmarks"] else [])))
    prices = snapshot_prices(md, tickers_for_report, asof)
    port_block, total_equity = compute_equity_block(md, positions, cash, asof, settings["report_benchmarks"])
    print_daily_report(asof, prices, port_block)

    return 0


if __name__ == "__main__":
    sys.exit(main())
