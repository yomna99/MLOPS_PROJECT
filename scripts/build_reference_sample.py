from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_PATH = Path("data/raw/AIML Dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/ref_data_sample.csv")
REFERENCE_COLUMNS = [
    "isFraud",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]


def build_reference_sample(
    *,
    input_path: Path,
    output_path: Path,
    rows: int,
    random_state: int,
) -> Path:
    frame = pd.read_csv(input_path, usecols=REFERENCE_COLUMNS)
    if frame.empty:
        raise ValueError(f"Reference source is empty: {input_path}")

    sample_size = min(rows, len(frame))
    if sample_size < len(frame):
        sampled, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame["isFraud"],
            random_state=random_state,
        )
    else:
        sampled = frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.sort_values(["step", "amount"], inplace=True)
    sampled.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight reference dataset sample for Evidently.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the raw fraud dataset CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the sampled reference CSV will be written.",
    )
    parser.add_argument("--rows", type=int, default=2500, help="Number of rows to keep in the sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_reference_sample(
        input_path=args.input,
        output_path=args.output,
        rows=args.rows,
        random_state=args.seed,
    )
    print(f"Reference sample written to {output_path}")


if __name__ == "__main__":
    main()
