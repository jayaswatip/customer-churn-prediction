from __future__ import annotations

from typing import Any

import pandas as pd


def quality_checks(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    activity_before_cutoff: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    schema = config["schema"]
    results: dict[str, pd.DataFrame] = {}
    missing_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for name, frame in tables.items():
        for column in frame.columns:
            missing_rows.append(
                {
                    "table": name,
                    "column": column,
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": float(frame[column].isna().mean() * 100),
                }
            )
        id_column = schema["customer_id"].get(name)
        if id_column and id_column in frame.columns:
            duplicate_rows.append(
                {
                    "table": name,
                    "duplicate_rows": int(frame[id_column].duplicated(keep=False).sum()),
                    "duplicate_customer_ids": int(frame[id_column].duplicated().sum()),
                }
            )
    results["missing_values"] = pd.DataFrame(missing_rows)
    results["duplicates"] = pd.DataFrame(duplicate_rows)

    activity_name = "activity"
    activity = tables[activity_name]
    month_column = schema["activity_month"]
    activity_id = schema["customer_id"][activity_name]
    activity_dates = pd.to_datetime(activity[month_column], errors="coerce")
    valid = activity.loc[activity_dates.notna()].copy()
    valid["_month"] = activity_dates.loc[activity_dates.notna()].dt.to_period("M")
    activity_duplicate_rows = int(
        valid.duplicated(subset=[activity_id, "_month"], keep=False).sum()
    )
    results["duplicate_customer_months"] = pd.DataFrame(
        [
            {
                "duplicate_customer_month_rows": activity_duplicate_rows,
                "duplicate_customer_month_keys": int(
                    valid.duplicated(subset=[activity_id, "_month"]).sum()
                ),
            }
        ]
    )
    cutoff_period = pd.Period(config["label_window"]["start"], freq="M")
    pre = valid.loc[valid["_month"] < cutoff_period]
    if pre.empty:
        global_missing = []
        customer_missing = []
    else:
        observed_months = set(pre["_month"].unique())
        expected_months = pd.period_range(min(observed_months), cutoff_period - 1, freq="M")
        global_missing = [
            {"missing_month": str(month)} for month in expected_months if month not in observed_months
        ]
        customer_missing = []
        for customer_id, rows in pre.groupby(activity_id):
            customer_months = set(rows["_month"].unique())
            expected = pd.period_range(min(customer_months), cutoff_period - 1, freq="M")
            absent = [str(month) for month in expected if month not in customer_months]
            if absent:
                customer_missing.append(
                    {"customer_id": customer_id, "missing_months": ", ".join(absent)}
                )
    results["missing_global_months"] = pd.DataFrame(global_missing, columns=["missing_month"])
    results["missing_customer_months"] = pd.DataFrame(
        customer_missing, columns=["customer_id", "missing_months"]
    )

    invalid_spend: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        for column in schema.get("spend_columns", []):
            if table_name == activity_name and column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                invalid_spend.append(
                    {
                        "table": table_name,
                        "column": column,
                        "negative_value_rows": int((values < 0).sum()),
                        "null_or_non_numeric_rows": int(values.isna().sum()),
                    }
                )
    results["invalid_spend"] = pd.DataFrame(invalid_spend)

    count_column = schema.get("transaction_count_column")
    if count_column and count_column in activity.columns:
        counts = pd.to_numeric(activity[count_column], errors="coerce")
        results["zero_transactions"] = pd.DataFrame(
            [
                {
                    "column": count_column,
                    "zero_rows": int((counts == 0).sum()),
                    "negative_rows": int((counts < 0).sum()),
                    "null_or_non_numeric_rows": int(counts.isna().sum()),
                }
            ]
        )
    else:
        results["zero_transactions"] = pd.DataFrame(
            [{"note": "Set schema.transaction_count_column to enable this check."}]
        )

    casing: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        for column in schema.get("categorical_columns", []):
            if column not in frame.columns:
                continue
            values = frame[column].dropna().astype(str)
            normalized = values.str.strip().str.casefold()
            casing.append(
                {
                    "table": table_name,
                    "column": column,
                    "observed_spellings": int(values.nunique()),
                    "casefolded_spellings": int(normalized.nunique()),
                    "inconsistent_case_candidates": int(values.nunique() - normalized.nunique()),
                }
            )
    results["inconsistent_casing"] = pd.DataFrame(casing)

    outliers: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        for column in frame.select_dtypes(include="number").columns:
            values = frame[column].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers.append(
                {
                    "table": table_name,
                    "column": column,
                    "outlier_rows_iqr": int(((values < lower) | (values > upper)).sum()),
                    "lower_fence": float(lower),
                    "upper_fence": float(upper),
                }
            )
    results["numeric_outliers_iqr"] = pd.DataFrame(outliers)
    return results


def write_quality_reports(
    reports: dict[str, pd.DataFrame], output_dir: str
) -> None:
    from pathlib import Path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(destination / f"quality_{name}.csv", index=False)
