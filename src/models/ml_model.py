"""
Machine Learning model training and prediction
"""
import xgboost as xgb
import pickle
import os
from sklearn.metrics import mean_absolute_error
from logs import logger


class MLModel:
    """XGBoost regression model for predicting price movements"""

    def __init__(self, ml_config, output_config, profile):
        """
        Initialize ML model

        Args:
            ml_config: ML configuration dictionary
            output_config: Output configuration dictionary
            profile: Profile name for saving
        """
        self.ml_config = ml_config
        self.output_config = output_config
        self.profile = profile
        self.model = None
        self.feature_cols = [
            "rsi",
            "bb_position",
            "volume_ratio",
            "volatility",
            "momentum_5",
            "momentum_10",
            "price_to_sma_fast",
            "price_to_sma_slow",
            "sma_ratio",
        ]

    def train(self, df_train):
        """
        Train the ML model on training data

        Args:
            df_train: Training DataFrame with indicators

        Returns:
            tuple: (model, feature_cols) or (None, None) if ML disabled
        """
        if not self.ml_config["enabled"]:
            logger.info("ML disabled for this profile")
            return None, None

        df_train = df_train.copy()

        # Create target: maximum gain in next N hours
        horizon = self.ml_config["prediction_horizon"]
        df_train["future_max"] = df_train["high"].rolling(horizon).max().shift(-horizon)
        df_train["max_gain_pct"] = (
            df_train["future_max"] - df_train["close"]
        ) / df_train["close"]
        df_train = df_train[:-horizon].copy()

        # Train only on SMA cross-up signals
        cross_up_samples = df_train[df_train["sma_cross_up"]].copy()

        if len(cross_up_samples) < 20:
            logger.warning(
                f"Only {len(cross_up_samples)} cross-up samples. ML may not work well."
            )
            return None, None

        logger.info(f"Training ML on {len(cross_up_samples)} SMA cross-up signals")
        logger.info(
            f"  Average max gain: {cross_up_samples['max_gain_pct'].mean() * 100:.2f}%"
        )

        # Prepare features and target
        X = cross_up_samples[self.feature_cols].fillna(0).values
        y = cross_up_samples["max_gain_pct"].values

        # Train model
        model_params = self.ml_config["model_params"]
        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X, y)

        # Evaluate
        train_pred = self.model.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        logger.info(f"  Training MAE: {train_mae * 100:.2f}%")

        # Save model
        if self.output_config.get("save_model", True):
            self._save_model()

        return self.model, self.feature_cols

    def predict(self, df):
        """
        Make predictions on new data

        Args:
            df: DataFrame with indicators

        Returns:
            np.array: Predictions or None if model not trained
        """
        if self.model is None:
            return None

        X = df[self.feature_cols].fillna(0).values
        return self.model.predict(X)

    def _save_model(self):
        """Save trained model to disk"""
        model_path = os.path.join(
            self.output_config["base_path"], f"ml_model_{self.profile}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_cols}, f)
        logger.info(f"  Model saved to {model_path}")
