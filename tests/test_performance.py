import unittest
import pandas as pd
from src.evaluation.performance_metrics import calculate_cumulative_return, calculate_sharpe_ratio, calculate_drawdowns, calculate_sortino_ratio

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        self.data = {
            'timestamp': ['2025-09-01 23:50:00', '2025-09-02 00:00:00', '2025-09-02 00:10:00'],
            'close': [108444.96, 108529.98, 108739.99]
        }
        self.df = pd.DataFrame(self.data)
        self.df['returns'] = self.df['close'].pct_change().dropna()

    def test_cumulative_return(self):
        cum_return = calculate_cumulative_return(self.df['close'])
        self.assertAlmostEqual(cum_return, (self.df['close'].iloc[-1] / self.df['close'].iloc[0]) - 1)

    def test_sharpe_ratio(self):
        sharpe_ratio = calculate_sharpe_ratio(self.df['returns'])
        self.assertIsInstance(sharpe_ratio, float)

    def test_drawdowns(self):
        drawdowns = calculate_drawdowns(self.df['close'])
        self.assertIsInstance(drawdowns, pd.Series)

    def test_sortino_ratio(self):
        sortino_ratio = calculate_sortino_ratio(self.df['returns'])
        self.assertIsInstance(sortino_ratio, float)

if __name__ == '__main__':
    unittest.main()