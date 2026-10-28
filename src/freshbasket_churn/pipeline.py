from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from freshbasket_churn.analysis import churn_relationships, save_eda_charts
from freshbasket_churn.config import resolve_path, validate_schema
from freshbasket_churn.data import (
    feature_columns,
    input_table_paths,
    prepare_customer_dataset,
    read_table,
    validate_feature_groups,
)
from freshbasket_churn.modeling import (
    build_feature_importance,
    evaluate_models,
    make_customer_predictions,
    retention_scenario,
)
from freshbasket_churn.quality import quality_checks, write_quality_reports
from freshbasket_churn.reporting import create_presentation, write_final_report


def _read_inputs(config: dict[str, Any], config_path: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    paths = input_table_paths(config, config_path)
    for name in ("profile", "activity", "labels"):
        tables[name] = read_table(paths[name])
    return tables


def run_pipeline(config: dict[str, Any], config_path: str | Path) -> None:
    validate_schema(config)
    validate_feature_groups(config)
    tables = _read_inputs(config, config_path)
    output_dir = resolve_path(config_path, config["paths"]["outputs"])
    processed_dir = resolve_path(config_path, config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    quality = quality_checks(tables, config)
    write_quality_reports(quality, str(output_dir))
    customers, _ = prepare_customer_dataset(tables, config)
    groups = feature_columns(config)
    clean_columns = list(
        dict.fromkeys(
            ["customer_id", "churn"]
            + [column for columns in groups.values() for column in columns]
        )
    )
    customers[clean_columns].to_csv(
        processed_dir / "cleaned_modeling_dataset.csv", index=False
    )

    numerical, categorical = churn_relationships(customers, config)
    numerical.to_csv(output_dir / "numeric_churn_relationships.csv", index=False)
    categorical.to_csv(output_dir / "categorical_churn_relationships.csv", index=False)

    results, best = evaluate_models(customers, config)
    best["dataset_label"] = config.get("dataset_label", "UNSPECIFIED DATA SOURCE")
    results.to_csv(output_dir / "validation_results.csv", index=False)
    importance = build_feature_importance(best, int(config.get("seed", 42)))
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    predictions = make_customer_predictions(best, importance)
    predictions.to_csv(output_dir / "customer_predictions.csv", index=False)
    scenario = retention_scenario(predictions, config)
    scenario.to_csv(output_dir / "retention_scenario.csv", index=False)
    save_eda_charts(customers, numerical, importance, output_dir)

    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "feature_columns": best["columns"],
            "model_name": best["model_name"],
            "validation_cohorts": best["validation_cohorts_note"],
            "feature_cutoff": config["label_window"]["start"],
            "seed": config.get("seed", 42),
        },
        output_dir / "trained_model.joblib",
    )
    write_final_report(
        customers, results, importance, scenario, config, output_dir / "final_report.md"
    )
    create_presentation(
        {
            "customers": customers,
            "results": results,
            "importance": importance,
            "scenario": scenario,
            "dataset_label": config.get("dataset_label", "UNSPECIFIED DATA SOURCE"),
        },
        output_dir / "retention_review_presentation.pptx",
    )
    print(f"Pipeline complete. Processed data: {processed_dir}")
    print(f"Generated analysis deliverables: {output_dir}")
