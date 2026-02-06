"""CLI wrapper for lithology train/predict so other languages can call it."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from geosc.well.ml import LithologyPredictor


def _parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON for parameters") from exc


def _parse_features(raw: str) -> list[str]:
    return [f.strip() for f in raw.split(",") if f.strip()]


def _mask_null_rows(X: np.ndarray, null_value: float | None) -> np.ndarray:
    if null_value is None:
        return X
    X = np.asarray(X, dtype=float)
    X[X == null_value] = np.nan
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Lithology prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a lithology model")
    train_parser.add_argument("--input", required=True, help="Input CSV file")
    train_parser.add_argument("--features", required=True, help="Comma-separated feature columns")
    train_parser.add_argument("--target", required=True, help="Target column")
    train_parser.add_argument("--model-type", default="mlp", help="Model type")
    train_parser.add_argument("--model-params", default=None, help="JSON string of model parameters")
    train_parser.add_argument("--scaler-params", default=None, help="JSON string of scaler parameters")
    train_parser.add_argument("--null-value", type=float, default=None, help="Treat this value as NULL")
    train_parser.add_argument("--output", required=True, help="Output model path (.pkl)")

    predict_parser = subparsers.add_parser("predict", help="Predict lithology using a trained model")
    predict_parser.add_argument("--input", required=True, help="Input CSV file")
    predict_parser.add_argument("--features", required=True, help="Comma-separated feature columns")
    predict_parser.add_argument("--model", required=True, help="Trained model path (.pkl)")
    predict_parser.add_argument("--output", required=True, help="Output CSV file")
    predict_parser.add_argument("--null-value", type=float, default=None, help="Treat this value as NULL")
    predict_parser.add_argument(
        "--output-null-value",
        type=float,
        default=None,
        help="Write this value for NULL predictions (default: NaN)",
    )
    predict_parser.add_argument("--probabilities", action="store_true", help="Include probability columns")

    args = parser.parse_args()

    if args.command == "train":
        df = pd.read_csv(args.input)
        features = _parse_features(args.features)
        X = df[features].values
        y = df[args.target].values

        X = _mask_null_rows(X, args.null_value)

        predictor = LithologyPredictor(model_type=args.model_type)
        predictor.train(
            X,
            y,
            parameters=_parse_params(args.model_params),
            scaler_params=_parse_params(args.scaler_params),
        )
        predictor.save(args.output)
        return

    if args.command == "predict":
        df = pd.read_csv(args.input)
        features = _parse_features(args.features)
        X = df[features].values

        X = _mask_null_rows(X, args.null_value)

        predictor = LithologyPredictor.load(args.model)
        pred, prob = predictor.predict(X, null_value=args.output_null_value)

        df_out = df.copy()
        df_out["lithology_pred"] = pred
        if args.probabilities and prob is not None:
            for idx in range(prob.shape[1]):
                df_out[f"lithology_prob_{idx}"] = prob[:, idx]

        df_out.to_csv(args.output, index=False)
        return


if __name__ == "__main__":
    main()
