from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freshbasket_churn.data import DataInputError, feature_columns


def temporal_split(
    customers: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.Index, pd.Index, str]:
    cohort_column = "signup_cohort"
    cohorts = sorted(customers[cohort_column].dropna().unique())
    holdout_count = int(config.get("validation", {}).get("holdout_latest_cohorts", 1))
    if holdout_count < 1 or len(cohorts) <= holdout_count:
        raise DataInputError(
            f"Temporal validation needs more than {holdout_count} distinct signup cohorts; "
            f"found {len(cohorts)}."
        )
    validation_cohorts = set(cohorts[-holdout_count:])
    validation_mask = customers[cohort_column].isin(validation_cohorts)
    train_idx = customers.index[~validation_mask]
    validation_idx = customers.index[validation_mask]
    if customers.loc[train_idx, "churn"].nunique() < 2:
        raise DataInputError("Training cohorts must contain both churn classes.")
    if customers.loc[validation_idx, "churn"].nunique() < 2:
        raise DataInputError(
            "The latest signup cohort(s) do not contain both classes. Adjust "
            "validation.holdout_latest_cohorts or collect more cohort data; random splitting is not used."
        )
    return train_idx, validation_idx, ", ".join(sorted(validation_cohorts))


def _selected_columns(config: dict[str, Any], selection: str) -> list[str]:
    groups = feature_columns(config)
    if selection == "demographics_only":
        chosen_groups = {"demographics"}
    elif selection == "behavior_engagement_support":
        chosen_groups = {"behavioral", "engagement", "support"}
    elif selection == "all_features":
        chosen_groups = set(groups)
    else:
        raise ValueError(f"Unknown feature selection: {selection}")
    columns = list(dict.fromkeys(
        column for group, values in groups.items() if group in chosen_groups for column in values
    ))
    if not columns:
        raise DataInputError(f"No mapped feature columns available for '{selection}'.")
    forbidden = set(config["schema"].get("forbidden_feature_columns", []))
    leaked = forbidden & set(columns)
    if leaked:
        raise DataInputError(f"Forbidden columns selected as features: {sorted(leaked)}.")
    return columns


def _build_pipeline(frame: pd.DataFrame, model_name: str, seed: int) -> Pipeline:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def evaluate_models(
    customers: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_idx, validation_idx, cohort_note = temporal_split(customers, config)
    y_train = customers.loc[train_idx, "churn"].astype(int)
    y_validation = customers.loc[validation_idx, "churn"].astype(int)
    records: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    selections = (
        "demographics_only",
        "behavior_engagement_support",
        "all_features",
    )
    for selection in selections:
        columns = _selected_columns(config, selection)
        x_train = customers.loc[train_idx, columns]
        x_validation = customers.loc[validation_idx, columns]
        for model_name in ("LogisticRegression", "RandomForest"):
            pipeline = _build_pipeline(x_train, model_name, int(config.get("seed", 42)))
            pipeline.fit(x_train, y_train)
            probabilities = pipeline.predict_proba(x_validation)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            result = {
                "model": model_name,
                "feature_set": selection,
                "validation_cohorts": cohort_note,
                "precision": precision_score(y_validation, predictions, zero_division=0),
                "recall": recall_score(y_validation, predictions, zero_division=0),
                "f1": f1_score(y_validation, predictions, zero_division=0),
                "pr_auc": average_precision_score(y_validation, probabilities),
                "accuracy": accuracy_score(y_validation, predictions),
                "validation_customers": len(validation_idx),
            }
            records.append(result)
            key = (model_name, selection)
            candidates[key] = {
                "pipeline": pipeline,
                "columns": columns,
                "validation_idx": validation_idx,
                "probabilities": probabilities,
                "predictions": predictions,
            }
    results = pd.DataFrame(records).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    all_features = results.loc[results["feature_set"] == "all_features"]
    best_model_name = all_features.iloc[0]["model"]
    best = candidates[(best_model_name, "all_features")]
    best["model_name"] = best_model_name
    best["validation_cohorts_note"] = cohort_note
    best["customers"] = customers
    best["y_validation"] = y_validation
    best["train_idx"] = train_idx
    return results, best


def build_feature_importance(best: dict[str, Any], seed: int) -> pd.DataFrame:
    validation_idx = best["validation_idx"]
    customers = best["customers"]
    columns = best["columns"]
    x_validation = customers.loc[validation_idx, columns]
    y_validation = best["y_validation"]
    result = permutation_importance(
        best["pipeline"],
        x_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": columns,
            "permutation_importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def make_customer_predictions(best: dict[str, Any], importance: pd.DataFrame) -> pd.DataFrame:
    customers = best["customers"]
    validation_idx = best["validation_idx"]
    columns = best["columns"]
    validation = customers.loc[validation_idx, ["customer_id", "churn", "signup_cohort"]].copy()
    probabilities = best["probabilities"]
    validation["churn_probability"] = probabilities
    validation["predicted_churn"] = best["predictions"]
    top_features = importance.loc[importance["permutation_importance"] > 0, "feature"].head(3).tolist()
    if not top_features:
        top_features = importance["feature"].head(3).tolist()
    driver_text: list[str] = []
    for index in validation_idx:
        cues = []
        for column in top_features:
            value = customers.at[index, column]
            if pd.isna(value):
                value = "(missing)"
            cues.append(f"{column}={value}")
        driver_text.append("; ".join(cues))
    validation["key_model_cues"] = driver_text
    validation["prediction_scope"] = "out_of_time_validation"
    validation["model"] = best["model_name"]
    validation["dataset_label"] = best.get("dataset_label", "UNSPECIFIED DATA SOURCE")
    return validation.reset_index(drop=True)


def retention_scenario(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    settings = config.get("retention_scenario", {})
    fraction = float(settings.get("contact_fraction", 0.10))
    lift = float(settings.get("assumed_absolute_retention_lift", 0.10))
    cost = float(settings.get("contact_cost_per_customer", 0.0))
    value = float(settings.get("assumed_value_per_retained_customer", 0.0))
    if not 0 <= fraction <= 1 or not 0 <= lift <= 1 or cost < 0 or value < 0:
        raise DataInputError("Retention scenario assumptions must be within valid non-negative bounds.")
    ranked = predictions.sort_values("churn_probability", ascending=False)
    contacted = min(len(ranked), math.ceil(len(ranked) * fraction))
    selected = ranked.head(contacted)
    expected_avoided = contacted * lift
    total_cost = contacted * cost
    estimated_value = expected_avoided * value
    return pd.DataFrame(
        [
            {
                "validation_customers": len(predictions),
                "contact_fraction_assumption": fraction,
                "customers_contacted": contacted,
                "assumed_absolute_retention_lift": lift,
                "estimated_incremental_retained_customers": expected_avoided,
                "assumed_contact_cost_per_customer": cost,
                "estimated_contact_cost": total_cost,
                "assumed_value_per_retained_customer": value,
                "estimated_value_of_retained_customers": estimated_value,
                "estimated_net_value": estimated_value - total_cost,
                "mean_predicted_churn_probability_contacted": (
                    float(selected["churn_probability"].mean()) if contacted else 0.0
                ),
                "causal_claim": "No; scenario is assumption-based and not a causal estimate.",
            }
        ]
    )
