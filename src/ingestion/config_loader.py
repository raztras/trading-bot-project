"""
Configuration loader for trading profiles
"""
import yaml
from logs import logger


class ConfigLoader:
    """Load and manage trading configuration profiles"""

    @staticmethod
    def load_profile(config_path="trading_profiles.yaml", profile="WINNER"):
        """
        Load a trading profile from YAML configuration

        Args:
            config_path: Path to the YAML config file
            profile: Profile name to load

        Returns:
            tuple: (profile_config, data_config, output_config)
        """
        logger.info(f"Loading config from {config_path}, profile: {profile}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if profile not in config["profiles"]:
            raise ValueError(
                f"Profile '{profile}' not found in config. Available: {list(config['profiles'].keys())}"
            )

        profile_config = config["profiles"][profile]
        data_config = config["data"]
        output_config = config["output"]

        logger.info(f"Profile: {profile}")
        logger.info(f"Description: {profile_config['description']}")

        return profile_config, data_config, output_config
