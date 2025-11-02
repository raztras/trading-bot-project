#!/bin/bash
# Raspberry Pi 5 Setup Script for Paper Trading
# Run this once on your Raspberry Pi to set up automated paper trading

set -e

echo "================================================"
echo "Trading Bot - Raspberry Pi Setup"
echo "================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "📁 Project directory: $SCRIPT_DIR"

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "❌ Virtual environment not found!"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
SCRIPT_PATH="$SCRIPT_DIR/run_paper_trading.py"

echo ""
echo "✅ Python: $PYTHON_PATH"
echo "✅ Script: $SCRIPT_PATH"

# Test the script
echo ""
echo "🧪 Testing paper trading script..."
$PYTHON_PATH "$SCRIPT_PATH" --profile OPTIMIZED || {
    echo "❌ Test failed! Fix errors before continuing."
    exit 1
}

echo ""
echo "✅ Test successful!"

# Set up cron job
echo ""
echo "📅 Setting up cron job..."
echo "   This will run paper trading at :05 past each hour"

# Create cron entry
CRON_ENTRY="5 * * * * cd $SCRIPT_DIR && $PYTHON_PATH $SCRIPT_PATH --profile OPTIMIZED >> $SCRIPT_DIR/paper_trading.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "run_paper_trading.py"; then
    echo "⚠️  Cron job already exists. Removing old entry..."
    crontab -l | grep -v "run_paper_trading.py" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✅ Cron job added:"
echo "   $CRON_ENTRY"

# Create log file
touch "$SCRIPT_DIR/paper_trading.log"
echo "✅ Log file created: $SCRIPT_DIR/paper_trading.log"

# Show current crontab
echo ""
echo "📋 Current cron jobs:"
crontab -l

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Paper trading will now run automatically at :05 past each hour"
echo ""
echo "Useful commands:"
echo "  View logs:        tail -f $SCRIPT_DIR/paper_trading.log"
echo "  Test manually:    $PYTHON_PATH $SCRIPT_PATH --profile OPTIMIZED"
echo "  Edit cron:        crontab -e"
echo "  Remove cron:      crontab -l | grep -v run_paper_trading.py | crontab -"
echo ""
echo "Next steps:"
echo "  1. Monitor first few runs: tail -f paper_trading.log"
echo "  2. Check Supabase for data"
echo "  3. Let it run!"
echo ""
