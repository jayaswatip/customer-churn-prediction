from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from freshbasket_churn.data import feature_columns


def churn_relationships(
    customers: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = feature_columns(config)
    numerical_rows: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    for group, columns in groups.items():
        for column in columns:
            if column not in customers.columns:
                continue
            series = customers[column]
            if pd.api.types.is_numeric_dtype(series):
                by_label = customers.groupby("churn")[column].mean()
                numerical_rows.append(
                    {
                        "feature": column,
                        "group": group,
                        "mean_not_churn": by_label.get(0),
                        "mean_churn": by_label.get(1),
                        "difference_churn_minus_not_churn": (
                            by_label.get(1, float("nan")) - by_label.get(0, float("nan"))
                        ),
                        "missing_count": int(series.isna().sum()),
                    }
                )
            else:
                data = customers[[column, "churn"]].copy()
                data[column] = data[column].fillna("(missing)").astype(str)
                summary = (
                    data.groupby(column, dropna=False)["churn"]
                    .agg(customers="size", churn_rate="mean")
                    .reset_index()
                    .rename(columns={column: "value"})
                )
                summary["feature"] = column
                summary["group"] = group
                category_frames.append(summary[["feature", "group", "value", "customers", "churn_rate"]])
    numerical = pd.DataFrame(numerical_rows)
    categorical = (
        pd.concat(category_frames, ignore_index=True)
        if category_frames
        else pd.DataFrame(columns=["feature", "group", "value", "customers", "churn_rate"])
    )
    return numerical, categorical


def save_eda_charts(
    customers: pd.DataFrame,
    numerical: pd.DataFrame,
    importance: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    churn_rates = customers.groupby("signup_cohort", observed=True)["churn"].agg(
        customers="size", churn_rate="mean"
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    churn_rates["churn_rate"].plot(kind="bar", ax=axis, color="#168a70")
    axis.set(title="Observed churn rate by signup cohort", xlabel="Signup cohort", ylabel="Churn rate")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(destination / "churn_by_signup_cohort.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    customers["churn"].value_counts(normalize=True).sort_index().plot(
        kind="bar", ax=axis, color=["#7aa6c2", "#db795c"]
    )
    axis.set(title="Churn label distribution", xlabel="Churn label", ylabel="Share of customers")
    axis.set_xticklabels(["Not churned", "Churned"], rotation=0)
    figure.tight_layout()
    figure.savefig(destination / "label_distribution.png", dpi=160)
    plt.close(figure)

    if not numerical.empty:
        selected = numerical.assign(
            magnitude=numerical["difference_churn_minus_not_churn"].abs()
        ).nlargest(12, "magnitude")
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=selected,
            x="difference_churn_minus_not_churn",
            y="feature",
            hue="group",
            dodge=False,
            ax=axis,
        )
        axis.set(title="Largest observed numeric mean differences by churn label", xlabel="Churn minus non-churn mean")
        figure.tight_layout()
        figure.savefig(destination / "numeric_churn_differences.png", dpi=160)
        plt.close(figure)

    if not importance.empty:
        selected = importance.head(15).sort_values("permutation_importance")
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.barplot(data=selected, x="permutation_importance", y="feature", color="#168a70", ax=axis)
        axis.set(title="Temporal-validation permutation importance (PR-AUC)", xlabel="Mean PR-AUC decrease")
        figure.tight_layout()
        figure.savefig(destination / "feature_importance.png", dpi=160)
        plt.close(figure)
