from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from freshbasket_churn.config import resolve_path


class DataInputError(ValueError):
    """Raised when an input table cannot safely be read or joined."""


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input table not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(source)
    raise DataInputError(f"Unsupported input format '{suffix}' for {source}. Use CSV or Excel.")


def input_table_paths(
    config: dict[str, Any], config_path: str | Path
) -> dict[str, Path]:
    paths = {
        name: resolve_path(config_path, config["paths"][name])
        for name in ("profile", "activity", "labels")
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        expected = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required source table file(s) not found:\n"
            f"{expected}\n"
            "Copy the supplied files to these locations or update their paths in "
            "config/project.yaml. Supported formats: .csv, .xlsx, .xls."
        )
    return paths


def inspect_tables(config: dict[str, Any], config_path: str | Path) -> dict[str, pd.DataFrame]:
    paths = input_table_paths(config, config_path)
    tables: dict[str, pd.DataFrame] = {}
    for name in ("profile", "activity", "labels"):
        frame = read_table(paths[name])
        tables[name] = frame
        print(f"\n[{name}] {len(frame):,} rows x {len(frame.columns):,} columns")
        print(f"Columns: {list(frame.columns)}")
        inventory = pd.DataFrame(
            {
                "dtype": frame.dtypes.astype(str),
                "missing": frame.isna().sum(),
                "missing_pct": (frame.isna().mean() * 100).round(2),
                "distinct": frame.nunique(dropna=True),
            }
        )
        print(inventory.to_string())
        print("Sample:")
        print(frame.head(3).to_string(index=False))
    return tables


def _require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise DataInputError(f"{table_name} is missing configured columns: {absent}")


def _check_unique_ids(frame: pd.DataFrame, id_column: str, table_name: str) -> None:
    if frame[id_column].isna().any():
        raise DataInputError(f"{table_name} contains null customer IDs in '{id_column}'.")
    duplicated = frame[id_column].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, id_column].drop_duplicates().head(5).tolist()
        raise DataInputError(
            f"{table_name} must have one row per customer; duplicate IDs include {examples}."
        )


def prepare_customer_dataset(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = config["schema"]
    customer_ids = schema["customer_id"]
    profile = tables["profile"].copy()
    activity = tables["activity"].copy()
    labels = tables["labels"].copy()
    profile_id = customer_ids["profile"]
    activity_id = customer_ids["activity"]
    label_id = customer_ids["labels"]
    month_column = schema["activity_month"]
    target_column = schema["target"]
    signup_column = schema["signup_date"]

    _require_columns(profile, [profile_id, signup_column, *schema["profile_features"]], "profile")
    _require_columns(
        activity,
        [activity_id, month_column, *schema["monthly_features"]],
        "activity",
    )
    _require_columns(labels, [label_id, target_column], "labels")
    _check_unique_ids(profile, profile_id, "profile")
    _check_unique_ids(labels, label_id, "labels")
    if labels[target_column].isna().any():
        raise DataInputError(f"Target column '{target_column}' contains missing values.")

    start = pd.Timestamp(config["label_window"]["start"])
    cutoff = start
    activity[month_column] = pd.to_datetime(activity[month_column], errors="coerce")
    invalid_months = activity[month_column].isna()
    if invalid_months.any():
        raise DataInputError(
            f"Activity has {int(invalid_months.sum())} invalid or missing values in '{month_column}'."
        )
    pre_window_rows = activity.loc[activity[month_column] < cutoff]
    duplicate_months = pre_window_rows.duplicated(
        subset=[activity_id, month_column], keep=False
    )
    if duplicate_months.any():
        examples = (
            pre_window_rows.loc[duplicate_months, [activity_id, month_column]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise DataInputError(
            "Monthly activity must have one row per customer per month before aggregation; "
            f"duplicate customer-month keys include {examples}."
        )
    activity["_activity_month"] = activity[month_column].dt.to_period("M")
    activity = activity.loc[activity[month_column] < cutoff].copy()

    monthly_config = schema["monthly_features"]
    monthly_output: dict[str, pd.Series] = {}
    for raw_column, options in monthly_config.items():
        if not isinstance(options, dict) or options.get("aggregation") not in {
            "sum", "mean", "max", "min", "last"
        }:
            raise DataInputError(
                f"Monthly feature '{raw_column}' needs aggregation: sum, mean, max, min, or last."
            )
        aggregation = options["aggregation"]
        grouped = activity.sort_values(month_column).groupby(activity_id)[raw_column]
        if aggregation == "last":
            monthly_output[raw_column] = grouped.last()
        else:
            monthly_output[raw_column] = getattr(grouped, aggregation)()
    monthly = pd.DataFrame(monthly_output)
    monthly.index.name = profile_id
    monthly = monthly.reset_index()

    profile = profile.rename(columns={profile_id: "customer_id"})
    monthly = monthly.rename(columns={activity_id: "customer_id"})
    labels = labels.rename(columns={label_id: "customer_id"})
    if profile_id != "customer_id" and profile_id in profile.columns:
        profile = profile.rename(columns={profile_id: "customer_id"})
    label_values = set(schema["label_values"]["churn"])
    non_churn_values = set(schema["label_values"]["not_churn"])
    observed_labels = set(labels[target_column].dropna().unique())
    if not observed_labels:
        raise DataInputError(f"Target column '{target_column}' contains no non-missing labels.")
    unexpected = observed_labels - label_values - non_churn_values
    if unexpected:
        raise DataInputError(
            f"Target has unmapped values {sorted(map(str, unexpected))}. "
            "Configure schema.label_values.churn and not_churn explicitly."
        )
    if label_values & non_churn_values:
        raise DataInputError("Churn and not_churn label values must not overlap.")
    if observed_labels - label_values - non_churn_values:
        raise DataInputError("Every observed target value must be mapped.")
    labels["churn"] = labels[target_column].isin(label_values).astype("int8")

    customer = profile.merge(monthly, on="customer_id", how="left", validate="one_to_one")
    customer = customer.merge(
        labels[["customer_id", "churn"]], on="customer_id", how="inner", validate="one_to_one"
    )
    customer[signup_column] = pd.to_datetime(customer[signup_column], errors="coerce")
    if customer[signup_column].isna().any():
        raise DataInputError(f"Profile signup date '{signup_column}' contains invalid or missing values.")
    customer = customer.loc[customer[signup_column] < cutoff].copy()
    if customer.empty:
        raise DataInputError(
            "No labeled customers signed up before the feature cutoff; there is no "
            "eligible pre-window modelling population."
        )
    customer["signup_cohort"] = customer[signup_column].dt.to_period("Q").astype(str)
    customer["signup_tenure_days_at_cutoff"] = (
        cutoff - customer[signup_column]
    ).dt.days.clip(lower=0)
    return customer, activity


def feature_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    schema = config["schema"]
    groups: dict[str, list[str]] = {}
    for column, group in schema["profile_features"].items():
        groups.setdefault(group, []).append(column)
    for column, options in schema["monthly_features"].items():
        if not isinstance(options, dict) or not options.get("group"):
            raise DataInputError(f"Monthly feature '{column}' must define a feature group.")
        groups.setdefault(options["group"], []).append(column)
    groups.setdefault("signup_cohort", []).append("signup_cohort")
    groups.setdefault("signup_tenure_days_at_cutoff", []).append(
        "signup_tenure_days_at_cutoff"
    )
    return groups


def validate_feature_groups(config: dict[str, Any]) -> None:
    allowed = {"demographics", "membership", "city", "engagement", "support", "behavioral"}
    schema = config["schema"]
    invalid_profile = {
        group for group in schema["profile_features"].values() if group not in allowed
    }
    invalid_monthly = {
        options.get("group")
        for options in schema["monthly_features"].values()
        if isinstance(options, dict) and options.get("group") not in allowed
    }
    if invalid_profile or invalid_monthly:
        raise DataInputError(
            f"Invalid feature group(s): {sorted(invalid_profile | invalid_monthly)}. "
            f"Use: {sorted(allowed)}."
        )
    schema = config["schema"]
    reserved = (
        set(schema["customer_id"].values())
        | {
            schema["activity_month"],
            schema["signup_date"],
            schema["target"],
            "customer_id",
            "churn",
            "signup_cohort",
            "signup_tenure_days_at_cutoff",
        }
        | set(schema.get("forbidden_feature_columns", []))
    )
    configured = set(schema["profile_features"]) | set(schema["monthly_features"])
    overlap = reserved & configured
    if overlap:
        raise DataInputError(
            f"Identifiers, target/date fields, or explicitly forbidden columns cannot be "
            f"model features: {sorted(overlap)}."
        )
    if "demographics" not in schema["profile_features"].values():
        raise DataInputError(
            "Map at least one pre-window profile column to the demographics feature group "
            "to run the requested ablation."
        )
    if not schema["monthly_features"]:
        raise DataInputError("Map at least one pre-window monthly activity feature.")
    if not {"behavioral", "engagement", "support"} & (
        set(schema["profile_features"].values())
        | {item.get("group") for item in schema["monthly_features"].values() if isinstance(item, dict)}
    ):
        raise DataInputError(
            "Map at least one behavioural, engagement, or support feature for the requested ablation."
        )
    forbidden = set(schema.get("forbidden_feature_columns", []))
    configured = set(schema["profile_features"]) | set(schema["monthly_features"])
    overlap = forbidden & configured
    if overlap:
        raise DataInputError(f"Forbidden columns are mapped as features: {sorted(overlap)}.")
