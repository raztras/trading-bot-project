# Paper Trading Setup Guide

## 🎯 Overview

This paper trading system runs your trading strategy in real-time using live market data, tracking all trades and portfolio performance in Supabase.

**Key Features:**
- ✅ Runs hourly at :05 past each hour
- ✅ Fully automated via cron job
- ✅ Uses your trained ML models
- ✅ Tracks positions, portfolio, and signals in Supabase
- ✅ Designed for Raspberry Pi 5

---

## 📦 What's Included

### Core Files
- **`run_paper_trading.py`** - Main executable script
- **`src/paper_trading/paper_engine.py`** - Trading logic
- **`src/paper_trading/paper_db.py`** - Supabase data layer
- **`raspberry_pi_setup.sh`** - Automated setup script
- **`paper_trading.log`** - Execution logs (created automatically)

### Database Tables (Supabase)
- **`paper_sessions`** - Track trading sessions
- **`paper_positions`** - All trades (open + closed)
- **`paper_portfolio`** - Equity snapshots over time
- **`paper_signals`** - Every signal generated

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have:
- ✅ Python 3.8+
- ✅ Virtual environment set up (`.venv/`)
- ✅ All dependencies installed
- ✅ Supabase credentials in `.env` file
- ✅ ML model trained (run a backtest first)

### 2. Test Locally (Windows/Mac)

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Run paper trading once
python run_paper_trading.py --profile OPTIMIZED

# Check output for errors
```

### 3. Deploy to Raspberry Pi

**Transfer files:**
```bash
# From your local machine
scp -r trading-bot-project pi@raspberrypi.local:~/
```

**SSH into Raspberry Pi:**
```bash
ssh pi@raspberrypi.local
cd ~/trading-bot-project
```

**Run setup script:**
```bash
chmod +x raspberry_pi_setup.sh
./raspberry_pi_setup.sh
```

This will:
- ✅ Test the paper trading script
- ✅ Set up cron job to run hourly
- ✅ Create log file
- ✅ Verify everything works

### 4. Monitor

```bash
# Watch logs in real-time
tail -f paper_trading.log

# Check cron is running
crontab -l

# View recent entries
tail -n 50 paper_trading.log
```

---

## ⚙️ Configuration

### Profile Selection

Change which strategy to run:
```bash
python run_paper_trading.py --profile WINNER
```

Or update the cron job:
```bash
crontab -e
# Change --profile OPTIMIZED to --profile WINNER
```

### Starting Cash

Default: $100,000

To change, edit `src/paper_trading/paper_db.py`:
```python
def create_session(self, profile: str, initial_cash: float = 100000.0):
    # Change 100000.0 to your desired amount
```

### Timing

Default: Runs at :05 past each hour (12:05, 13:05, 14:05, etc.)

To change:
```bash
crontab -e
# Modify the cron schedule
# 5 * * * * = :05 past each hour
# */30 * * * * = Every 30 minutes
# 0 */4 * * * = Every 4 hours at :00
```

---

## 📊 Viewing Results

### Supabase Dashboard

**Query your data:**

```sql
-- Current portfolio state
SELECT * FROM paper_portfolio
WHERE profile = 'OPTIMIZED'
ORDER BY timestamp DESC
LIMIT 1;

-- All closed trades
SELECT
  entry_time,
  exit_time,
  entry_price,
  exit_price,
  pnl,
  pnl_pct,
  exit_reason
FROM paper_positions
WHERE status = 'closed'
ORDER BY exit_time DESC;

-- Performance summary
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
  ROUND(AVG(pnl_pct), 2) as avg_return_pct,
  ROUND(SUM(pnl), 2) as total_pnl
FROM paper_positions
WHERE status = 'closed' AND profile = 'OPTIMIZED';

-- Equity curve
SELECT timestamp, total_equity
FROM paper_portfolio
WHERE profile = 'OPTIMIZED'
ORDER BY timestamp;

-- Recent signals
SELECT * FROM paper_signals
ORDER BY timestamp DESC
LIMIT 20;
```

### Log Files

```bash
# View entire log
cat paper_trading.log

# View last 50 lines
tail -n 50 paper_trading.log

# Watch live (Ctrl+C to exit)
tail -f paper_trading.log

# Search for errors
grep -i error paper_trading.log

# Search for trades
grep -i "ENTERED\|EXITED" paper_trading.log
```

---

## 🐛 Troubleshooting

### Script not running

**Check cron is active:**
```bash
sudo systemctl status cron  # Ubuntu/Debian
```

**Check cron logs:**
```bash
grep CRON /var/log/syslog  # Ubuntu/Debian
```

**Test manually:**
```bash
cd ~/trading-bot-project
.venv/bin/python run_paper_trading.py --profile OPTIMIZED
```

### No ML predictions

**Train a model first:**
```bash
cd src
../.venv/bin/python main.py --profile OPTIMIZED --days 360
```

This creates `output/ml_model_OPTIMIZED.pkl`

### Supabase connection errors

**Check .env file exists:**
```bash
cat .env
# Should show sb_db and sb_api
```

**Test connection:**
```bash
python test_supabase.py
```

### Wrong time zone

**Check Raspberry Pi timezone:**
```bash
timedatectl

# Set timezone
sudo timedatectl set-timezone America/New_York
```

---

## 📈 Expected Behavior

### Hourly Execution (at :05)

1. Fetches latest market data (30 days for indicators)
2. Calculates technical indicators (SMA, RSI, etc.)
3. Gets ML prediction (if model exists)
4. Checks for trading signals:
   - **Buy**: SMA cross up + ML filter passes
   - **Sell**: Profit target, stop loss, SMA cross down, or max hold
5. Executes trades (if signal present)
6. Updates portfolio snapshot
7. Logs everything to Supabase

### First Run

```
✅ Creates new session in paper_sessions
✅ Initializes portfolio with $100,000 cash
✅ Waits for first buy signal
```

### During Position

```
📊 Monitors exit conditions every hour
💰 Tracks unrealized P&L
🔴 Exits when conditions met
```

### After Exit

```
✅ Records trade in paper_positions
💰 Updates portfolio with realized P&L
⚪ Waits for next buy signal
```

---

## 🛑 Stopping Paper Trading

### Temporary Stop

```bash
# Comment out the cron job
crontab -e
# Add # at the beginning of the line
# #5 * * * * cd ...

# Or remove entirely
crontab -l | grep -v run_paper_trading.py | crontab -
```

### End Session

Mark session as stopped in Supabase:
```sql
UPDATE paper_sessions
SET status = 'stopped', ended_at = NOW()
WHERE profile = 'OPTIMIZED' AND status = 'active';
```

---

## 📝 Best Practices

1. **Monitor first 24 hours** - Watch logs to ensure everything works
2. **Check weekly** - Review performance in Supabase
3. **Keep logs clean** - Rotate logs monthly to save space
4. **Backup .env** - Keep credentials secure
5. **Test changes locally** - Before deploying to Raspberry Pi
6. **Update ML model** - Re-train periodically with new data

---

## 🔐 Security Notes

- ✅ Never commit `.env` file to git
- ✅ Restrict access to Raspberry Pi
- ✅ Use read-only API keys when possible
- ✅ Monitor for unusual activity

---

## 📞 Support

**Check logs first:**
```bash
tail -f paper_trading.log
```

**Common issues:**
- ML model not found → Run backtest first
- Supabase errors → Check credentials in .env
- No trades happening → Check strategy triggers in logs
- Cron not running → Check cron service status

---

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Log file shows hourly executions
- ✅ `paper_portfolio` table updates every hour
- ✅ `paper_signals` logs every check
- ✅ Trades appear in `paper_positions` when signals trigger
- ✅ Portfolio equity changes over time

**Happy paper trading! 🚀**
