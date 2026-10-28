from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ProjectConfigError(ValueError):
    """Raised when project configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ProjectConfigError("Configuration must contain a YAML mapping.")
    return config


def resolve_path(config_path: str | Path, configured_path: str | Path) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    return Path(config_path).resolve().parent.parent / configured


def validate_schema(config: dict[str, Any]) -> None:
    schema = config.get("schema", {})
    required = ("customer_id", "activity_month", "signup_date", "target")
    missing = [key for key in required if not schema.get(key)]
    if missing:
        raise ProjectConfigError(
            "Map the actual dataset columns in config/project.yaml. "
            f"Missing schema values: {', '.join(missing)}. "
            "Run `freshbasket-churn inspect` first."
        )
    ids = schema["customer_id"]
    missing_ids = [table for table in ("profile", "activity", "labels") if not ids.get(table)]
    if missing_ids:
        raise ProjectConfigError(
            f"Map customer_id for these tables: {', '.join(missing_ids)}."
        )
    label_values = schema.get("label_values", {})
    if not label_values.get("churn") or not label_values.get("not_churn"):
        raise ProjectConfigError(
            "Set schema.label_values.churn and schema.label_values.not_churn "
            "to the observed target values."
        )
    if not isinstance(schema.get("profile_features"), dict):
        raise ProjectConfigError("schema.profile_features must be a mapping.")
    if not isinstance(schema.get("monthly_features"), dict):
        raise ProjectConfigError("schema.monthly_features must be a mapping.")
