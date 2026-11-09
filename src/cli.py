"""Command-line interface for Rail ETA.

Usage::

    python -m src.rail_eta.cli train
    python -m src.rail_eta.cli predict --distance 500 --stations 10
    python -m src.rail_eta.cli predict-batch --input routes.csv
"""

from __future__ import annotations

import argparse
import sys

from config import settings
from logger import setup_logging, get_logger
from pipeline import run_pipeline
from predictor import load_model, predict_batch, predict_single, format_duration

logger = get_logger(__name__)


def _cmd_train(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        run_pipeline()
        logger.info("Training pipeline completed successfully")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        return 1


def _cmd_predict(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        model = load_model()
        if args.batch:
            out = predict_batch(model, args.input, args.output)
            print(f"Batch predictions saved to {out}")
            return 0
        duration = predict_single(model, args.distance, args.stations)
        print(format_duration(duration))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rail-eta",
        description="Rail ETA - Train Journey Duration Prediction",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model pipeline")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Predict journey duration")
    p_pred.add_argument("--distance", type=float, required=True, help="Total distance in km")
    p_pred.add_argument("--stations", type=int, required=True, help="Number of stations")
    p_pred.add_argument("--batch", action="store_true", default=False, help=argparse.SUPPRESS)
    p_pred.set_defaults(func=_cmd_predict)

    p_batch = sub.add_parser("predict-batch", help="Batch predict from CSV")
    p_batch.add_argument("--input", required=True, help="Input CSV path")
    p_batch.add_argument("--output", default=None, help="Output CSV path")
    p_batch.set_defaults(func=lambda args: _cmd_predict(argparse.Namespace(
        batch=True, input=args.input, output=args.output, distance=None, stations=None
    )))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
