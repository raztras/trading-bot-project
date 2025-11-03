-- ==========================================
-- Supabase Schema for Trading Bot
-- Complete database schema for backtesting and paper trading
-- Run this in your Supabase SQL Editor
-- ==========================================

-- ==========================================
-- PART 1: BACKTESTING TABLES
-- ==========================================

-- 1. TRADES TABLE (Backtest Results)
-- Stores completed trades from backtesting runs
CREATE TABLE IF NOT EXISTS trades (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  run_timestamp TIMESTAMP NOT NULL,
  entry_time TIMESTAMP NOT NULL,
  exit_time TIMESTAMP NOT NULL,
  entry_price DECIMAL(12,2) NOT NULL,
  exit_price DECIMAL(12,2) NOT NULL,
  pnl DECIMAL(12,4) NOT NULL,
  pnl_pct DECIMAL(8,4) NOT NULL,
  bars_held INTEGER NOT NULL,
  exit_reason VARCHAR(50) NOT NULL,
  ml_predicted_gain DECIMAL(8,4),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_profile ON trades(profile);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_run_timestamp ON trades(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);

-- 2. PERFORMANCE SUMMARY TABLE (Backtest Metrics)
-- Stores aggregate metrics from each backtest run
CREATE TABLE IF NOT EXISTS performance_summary (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  run_timestamp TIMESTAMP NOT NULL,
  days_tested INTEGER NOT NULL,
  test_bars INTEGER NOT NULL,
  net_return_pct DECIMAL(8,4),
  annualized_return_pct DECIMAL(8,4),
  total_trades INTEGER,
  win_rate_pct DECIMAL(8,4),
  sharpe_ratio DECIMAL(8,4),
  max_drawdown_pct DECIMAL(8,4),
  profit_factor DECIMAL(8,4),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_profile ON performance_summary(profile);
CREATE INDEX IF NOT EXISTS idx_performance_run_timestamp ON performance_summary(run_timestamp);

-- 3. CURRENT SIGNALS TABLE (Live Monitoring)
-- Stores current market state for real-time dashboards
CREATE TABLE IF NOT EXISTS current_signals (
  symbol VARCHAR(20) PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  close_price DECIMAL(12,2),
  sma_fast DECIMAL(12,2),
  sma_slow DECIMAL(12,2),
  rsi DECIMAL(8,2),
  in_position BOOLEAN DEFAULT FALSE,
  ml_prediction DECIMAL(8,4),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_profile ON current_signals(profile);

-- ==========================================
-- PART 2: PAPER TRADING TABLES
-- ==========================================

-- 4. PAPER SESSIONS TABLE
-- Tracks paper trading sessions (multiple runs)
CREATE TABLE IF NOT EXISTS paper_sessions (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  session_name VARCHAR(100),

  -- Session info
  started_at TIMESTAMP NOT NULL,
  ended_at TIMESTAMP,
  status VARCHAR(20) DEFAULT 'active',    -- 'active', 'paused', 'stopped'

  -- Starting conditions
  initial_cash DECIMAL(12,2) DEFAULT 100000,

  -- Final metrics (populated when session ends)
  final_equity DECIMAL(12,2),
  total_return_pct DECIMAL(8,4),
  total_trades INTEGER,
  win_rate_pct DECIMAL(8,4),
  sharpe_ratio DECIMAL(8,4),
  max_drawdown_pct DECIMAL(8,4),

  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_sessions_profile ON paper_sessions(profile);
CREATE INDEX IF NOT EXISTS idx_paper_sessions_status ON paper_sessions(status);

-- 5. PAPER POSITIONS TABLE
-- Tracks all paper trading positions (open and closed)
CREATE TABLE IF NOT EXISTS paper_positions (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  symbol VARCHAR(20) NOT NULL,

  -- Entry details
  entry_time TIMESTAMP NOT NULL,
  entry_price DECIMAL(12,2) NOT NULL,
  position_size DECIMAL(12,8) NOT NULL,   -- BTC amount
  entry_cash DECIMAL(12,2) NOT NULL,      -- USD spent

  -- Exit details (NULL if still open)
  exit_time TIMESTAMP,
  exit_price DECIMAL(12,2),
  exit_cash DECIMAL(12,2),

  -- P&L
  pnl DECIMAL(12,4),
  pnl_pct DECIMAL(8,4),

  -- Metadata
  status VARCHAR(20) DEFAULT 'open',      -- 'open', 'closed'
  exit_reason VARCHAR(50),
  ml_predicted_gain DECIMAL(8,4),
  bars_held INTEGER,

  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_paper_positions_profile ON paper_positions(profile);
CREATE INDEX IF NOT EXISTS idx_paper_positions_symbol ON paper_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_positions_entry_time ON paper_positions(entry_time);

-- 6. PAPER PORTFOLIO TABLE
-- Tracks portfolio equity over time (snapshots)
CREATE TABLE IF NOT EXISTS paper_portfolio (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  timestamp TIMESTAMP NOT NULL,

  -- Portfolio state
  cash_balance DECIMAL(12,2) NOT NULL,
  position_value DECIMAL(12,2) NOT NULL,  -- Current value of open position
  total_equity DECIMAL(12,2) NOT NULL,

  -- Position info
  in_position BOOLEAN DEFAULT FALSE,
  current_symbol VARCHAR(20),
  position_entry_price DECIMAL(12,2),
  position_size DECIMAL(12,8),
  unrealized_pnl DECIMAL(12,4),

  -- Performance metrics (lifetime)
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  total_pnl DECIMAL(12,4) DEFAULT 0,

  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_portfolio_profile ON paper_portfolio(profile);
CREATE INDEX IF NOT EXISTS idx_paper_portfolio_timestamp ON paper_portfolio(timestamp);

-- 7. PAPER SIGNALS TABLE
-- Historical log of all trading signals generated
CREATE TABLE IF NOT EXISTS paper_signals (
  id BIGSERIAL PRIMARY KEY,
  profile VARCHAR(50) NOT NULL,
  symbol VARCHAR(20) NOT NULL,
  timestamp TIMESTAMP NOT NULL,

  -- Signal data
  signal_type VARCHAR(20) NOT NULL,       -- 'buy', 'sell', 'hold'
  close_price DECIMAL(12,2) NOT NULL,
  sma_fast DECIMAL(12,2),
  sma_slow DECIMAL(12,2),
  rsi DECIMAL(8,2),
  ml_prediction DECIMAL(8,4),

  -- Action taken
  action_taken VARCHAR(20),               -- 'entered', 'exited', 'ignored', 'none'
  reason TEXT,

  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_signals_profile ON paper_signals(profile);
CREATE INDEX IF NOT EXISTS idx_paper_signals_timestamp ON paper_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_signals_type ON paper_signals(signal_type);

-- ==========================================
-- PART 3: ANALYTICS VIEWS
-- ==========================================

-- View: Backtest Trade Analytics
CREATE OR REPLACE VIEW trade_analytics AS
SELECT
  profile,
  DATE_TRUNC('day', entry_time) as trade_date,
  COUNT(*) as num_trades,
  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
  ROUND(AVG(pnl_pct) * 100, 2) as avg_return_pct,
  ROUND(SUM(pnl), 2) as total_pnl,
  exit_reason
FROM trades
GROUP BY profile, trade_date, exit_reason
ORDER BY trade_date DESC;

-- View: Paper Trading Performance Summary
CREATE OR REPLACE VIEW paper_trading_summary AS
SELECT
  p.profile,
  COUNT(CASE WHEN p.status = 'closed' THEN 1 END) as total_trades,
  SUM(CASE WHEN p.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
  ROUND(AVG(CASE WHEN p.status = 'closed' THEN p.pnl_pct END), 2) as avg_return_pct,
  ROUND(SUM(CASE WHEN p.status = 'closed' THEN p.pnl ELSE 0 END), 2) as total_pnl,
  MAX(port.total_equity) as current_equity,
  COUNT(CASE WHEN p.status = 'open' THEN 1 END) as open_positions
FROM paper_positions p
LEFT JOIN LATERAL (
  SELECT total_equity
  FROM paper_portfolio
  WHERE profile = p.profile
  ORDER BY timestamp DESC
  LIMIT 1
) port ON true
GROUP BY p.profile;

-- View: Recent Paper Trading Activity
CREATE OR REPLACE VIEW paper_recent_activity AS
SELECT
  profile,
  timestamp,
  signal_type,
  action_taken,
  close_price,
  ml_prediction,
  reason
FROM paper_signals
ORDER BY timestamp DESC
LIMIT 50;

-- ==========================================
-- PART 4: DISABLE ROW LEVEL SECURITY
-- ==========================================
-- Required for API key access without RLS policies

ALTER TABLE trades DISABLE ROW LEVEL SECURITY;
ALTER TABLE performance_summary DISABLE ROW LEVEL SECURITY;
ALTER TABLE current_signals DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_positions DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_portfolio DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_signals DISABLE ROW LEVEL SECURITY;

-- ==========================================
-- PART 5: EXAMPLE QUERIES
-- ==========================================

-- Verify tables were created
-- SELECT schemaname, tablename, rowsecurity
-- FROM pg_tables
-- WHERE tablename IN ('trades', 'performance_summary', 'current_signals',
--                     'paper_sessions', 'paper_positions', 'paper_portfolio', 'paper_signals');

-- Check backtest data
-- SELECT * FROM trades ORDER BY created_at DESC LIMIT 10;
-- SELECT * FROM performance_summary ORDER BY run_timestamp DESC LIMIT 5;
-- SELECT * FROM trade_analytics LIMIT 20;

-- Check paper trading data
-- SELECT * FROM paper_sessions WHERE status = 'active';
-- SELECT * FROM paper_positions WHERE status = 'open';
-- SELECT * FROM paper_portfolio ORDER BY timestamp DESC LIMIT 10;
-- SELECT * FROM paper_trading_summary;
-- SELECT * FROM paper_recent_activity;

-- Current portfolio state
-- SELECT * FROM paper_portfolio
-- WHERE profile = 'OPTIMIZED'
-- ORDER BY timestamp DESC
-- LIMIT 1;

-- Equity curve (for charting)
-- SELECT timestamp, total_equity
-- FROM paper_portfolio
-- WHERE profile = 'OPTIMIZED'
-- ORDER BY timestamp;

-- All closed trades
-- SELECT entry_time, exit_time, entry_price, exit_price, pnl, pnl_pct, exit_reason
-- FROM paper_positions
-- WHERE status = 'closed' AND profile = 'OPTIMIZED'
-- ORDER BY exit_time DESC;

-- Win rate analysis
-- SELECT
--   COUNT(*) as total_trades,
--   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
--   ROUND(AVG(pnl_pct), 2) as avg_return_pct,
--   ROUND(SUM(pnl), 2) as total_pnl
-- FROM paper_positions
-- WHERE status = 'closed' AND profile = 'OPTIMIZED';

-- Recent signals log
-- SELECT timestamp, signal_type, action_taken, close_price, reason
-- FROM paper_signals
-- WHERE profile = 'OPTIMIZED'
-- ORDER BY timestamp DESC
-- LIMIT 20;
