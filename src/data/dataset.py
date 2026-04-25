from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/AIML Dataset.csv")


@dataclass(frozen=True)
class FraudDatasetConfig:
    """Configuration used to load the raw fraud dataset."""

    csv_path: Path = DEFAULT_DATA_PATH
    sample_size: int | None = 250_000
    random_state: int = 42


def load_fraud_dataset(config: FraudDatasetConfig) -> pd.DataFrame:
    """Load the raw dataset and optionally down-sample it for faster iteration."""
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if config.sample_size is not None and len(dataframe) > config.sample_size:
        dataframe = dataframe.sample(
            n=config.sample_size,
            random_state=config.random_state,
        )

    return dataframe.reset_index(drop=True)
