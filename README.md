# ChatGPT BIST Trading Experiment 🇹🇷

This project adapts the [ChatGPT Micro-Cap Experiment](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment) to the **Turkish stock market (Borsa İstanbul – BIST)**.

The idea: let **ChatGPT** act as an AI portfolio manager for a curated universe of Turkish stocks.  
Each day, the script generates a **daily report** with prices, volumes, and portfolio state.  
ChatGPT (or another LLM) then issues **BUY/SELL/HOLD/STOP instructions**, which the script executes and logs.  

---

## ✨ Features
- **BIST Integration**: Works with `.IS` tickers via Yahoo Finance (e.g., `THYAO.IS`, `AKBNK.IS`).
- **Curated Universes**:
  - Core (stable, liquid mid/large caps)
  - Satellite (smaller, speculative plays)
  - Benchmarks (`XU100.IS`, `USDTRY=X`)
- **Portfolio Tracking**: TRY-denominated cash & positions saved in `portfolio.csv`.
- **Trade Logging**: Every trade (BUY, SELL, AUTO_SELL, STOP) recorded in `trades.csv` with universe tag (CORE / SATELLITE / BENCHMARK).
- **Equity Tracking**: `equity.csv` logs daily portfolio value split by Core vs Satellite.
- **Stop-Loss Automation**: Supports trailing % stops and hard price floors.
- **Daily Reports**: Console report with Core, Satellite, and Benchmark tables.

---

## 📂 Repo Structure
```bash
.
├── trading_script.py   # main trading engine  
├── portfolio.csv       # active positions (auto-created)  
├── trades.csv          # trade history (auto-created)  
├── equity.csv          # equity history (auto-created)  
├── cash_log.csv        # records all cash deposits  (auto-created)  
├── daily_results/      # daily price snapshots  
├── demo_data/          # example portfolio, trades & reports (safe to commit)  
├── settings.json       # config (cash, stop % etc.)  
└── orders.txt          # ChatGPT trading plan (input)  
```
---

## ⚙️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/BeyzaOzkara/ChatGPT-BIST-Trading-Experiment.git
   cd ChatGPT-BIST-Trading-Experiment
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy yfinance matplotlib
   ```

3. (Optional) Edit `settings.json` to configure:
   ```json
   {
      "starting_cash": 1000,
      "default_trailing_stop_pct": 0.12,
      "commission_per_trade": 0.0,
      "slippage_bps": 0,
      "report_benchmarks": true,
      "core_universe": ["AKBNK.IS", "GARAN.IS", "ISCTR.IS"],
      "satellite_universe": ["KONTR.IS", "PENTA.IS"],
      "benchmarks": ["XU100.IS", "USDTRY=X"]
   }
   ```

---

## 🚀 Usage

### Generate Daily Report
```bash
python trading_script.py --mode report
```
Outputs:
- Daily price & volume tables (Core, Satellite, Benchmarks)
- Portfolio snapshot
- ChatGPT request prompt

### Apply Trading Plan
1. Copy report into ChatGPT, ask for trading instructions.  
2. Save them into `orders.txt`, for example:
   ```
   BUY THYAO.IS 20 @ MKT
   BUY ASELS.IS 10 @ MKT
   STOP THYAO.IS 12%
   HARDSTOP ASELS.IS 80.00
   ```
3. Apply:
   ```bash
   python trading_script.py --mode apply --orders orders.txt
   ```

### Stop-Loss Only Check
Run without new orders, just trailing stops:
```bash
python trading_script.py --mode stops-only
```

### Add New Cash
Deposit funds into the trading account for increased capital:
```bash
python trading_script.py --mode deposit --amount 5000 --note "..."
```

---

## 📊 Data Logging
- **Trades** → `trades.csv`  
- **Equity** → `equity.csv` (total, core, satellite, cash)  
- **Prices** → `daily_results/YYYY-MM-DD_prices.csv`  
- **Cash Deposits** → `cash_log.csv`
- **Demo Data** → `demo_data/`  

---

## 🔮 Roadmap
- [ ] Automate ticker universe via `settings.json`  
- [ ] Add visualization (equity curve, core vs satellite P&L)  
- [ ] Prompt/response journaling (journal/) for ChatGPT interactions  
- [ ] Weekly journal summaries  
- [ ] Weekly research mode (--mode weekly) with sector analysis & top movers
- [ ] Broker API integration for live execution  

---

## ⚠️ Disclaimer
This project is for **educational & experimental purposes only**.  
It does **not** constitute financial advice.  
Trading stocks involves risk. Use at your own discretion.  

---

## 🙌 Credits
- Original concept: [LuckyOne7777 / ChatGPT-Micro-Cap-Experiment](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)  
- Turkish market adaptation: this repo  
