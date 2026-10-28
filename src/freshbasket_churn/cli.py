from __future__ import annotations

import argparse
from pathlib import Path

from freshbasket_churn.config import load_config, resolve_path
from freshbasket_churn.data import inspect_tables
from freshbasket_churn.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="freshbasket-churn",
        description="Inspect FreshBasket churn tables or run the configured analysis pipeline.",
    )
    parser.add_argument(
        "command",
        choices=("inspect", "run"),
        help="'inspect' prints actual input columns; 'run' creates the analysis deliverables.",
    )
    parser.add_argument(
        "--config",
        default="config/project.yaml",
        help="Path to YAML configuration (default: config/project.yaml).",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    try:
        if args.command == "inspect":
            inspect_tables(config, config_path)
            return
        run_pipeline(config, config_path)
    except FileNotFoundError as error:
        parser.exit(status=2, message=f"Error: {error}\n")


if __name__ == "__main__":
    main()
