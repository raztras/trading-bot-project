import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.evaluation.performance_metrics import calculate_performance_metrics

class TradingEngine:
    def __init__(self, data, ma_short_period=3, ma_long_period=10):
        self.data = data
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.model = RandomForestClassifier()
        self.prepare_data()

    def prepare_data(self):
        self.data['ma_short'] = self.data['close'].rolling(window=self.ma_short_period).mean()
        self.data['ma_long'] = self.data['close'].rolling(window=self.ma_long_period).mean()
        self.data['signal'] = np.where(self.data['ma_short'] > self.data['ma_long'], 1, 0)
        self.data.dropna(inplace=True)

    def train_model(self):
        features = self.data[['ma_short', 'ma_long', 'plus1_std', 'minus1_std']]
        target = self.data['signal']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model Accuracy: {accuracy:.2f}')

    def execute_trades(self):
        self.data['predicted_signal'] = self.model.predict(self.data[['ma_short', 'ma_long', 'plus1_std', 'minus1_std']])
        self.data['returns'] = self.data['close'].pct_change()
        self.data['strategy_returns'] = self.data['predicted_signal'].shift(1) * self.data['returns']
        performance_metrics = calculate_performance_metrics(self.data['strategy_returns'])
        print(performance_metrics)

    def run(self):
        self.train_model()
        self.execute_trades()