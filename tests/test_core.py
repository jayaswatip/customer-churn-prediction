import pandas as pd
import pytest

from freshbasket_churn.modeling import temporal_split, retention_scenario
from freshbasket_churn.data import prepare_customer_dataset
from freshbasket_churn.data import input_table_paths


def test_temporal_split_holds_out_latest_cohort():
    customers = pd.DataFrame(
        {
            "signup_cohort": ["2023Q1", "2023Q1", "2023Q2", "2023Q2", "2023Q3", "2023Q3"],
            "churn": [0, 1, 0, 1, 0, 1],
        }
    )
    train, validation, cohort_note = temporal_split(
        customers, {"validation": {"holdout_latest_cohorts": 1}}
    )
    assert set(customers.loc[train, "signup_cohort"]) == {"2023Q1", "2023Q2"}
    assert set(customers.loc[validation, "signup_cohort"]) == {"2023Q3"}
    assert cohort_note == "2023Q3"


def test_retention_scenario_applies_explicit_assumptions():
    predictions = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "churn_probability": [0.9, 0.7, 0.2, 0.1],
        }
    )
    config = {
        "retention_scenario": {
            "contact_fraction": 0.5,
            "assumed_absolute_retention_lift": 0.2,
            "contact_cost_per_customer": 3,
            "assumed_value_per_retained_customer": 10,
        }
    }
    result = retention_scenario(predictions, config).iloc[0]
    assert result["customers_contacted"] == 2
    assert result["estimated_incremental_retained_customers"] == 0.4
    assert result["estimated_net_value"] == -2.0
    assert result["causal_claim"].startswith("No;")


def test_customer_dataset_uses_only_pre_window_activity():
    tables = {
        "profile": pd.DataFrame(
            {"member_key": ["a"], "joined_on": ["2023-01-15"], "age": [34]}
        ),
        "activity": pd.DataFrame(
            {
                "member_key": ["a", "a"],
                "activity_month": ["2024-03-01", "2024-04-01"],
                "spend": [25.0, 900.0],
            }
        ),
        "labels": pd.DataFrame({"member_key": ["a"], "outcome": ["churned"]}),
    }
    config = {
        "schema": {
            "customer_id": {
                "profile": "member_key",
                "activity": "member_key",
                "labels": "member_key",
            },
            "activity_month": "activity_month",
            "signup_date": "joined_on",
            "target": "outcome",
            "label_values": {"churn": ["churned"], "not_churn": ["active"]},
            "profile_features": {"age": "demographics"},
            "monthly_features": {
                "spend": {"group": "behavioral", "aggregation": "sum"}
            },
        },
        "label_window": {"start": "2024-04-01", "end": "2024-06-30"},
    }
    customers, pre_window_activity = prepare_customer_dataset(tables, config)
    assert customers.loc[0, "spend"] == 25.0
    assert customers.loc[0, "churn"] == 1
    assert pre_window_activity["activity_month"].max() < pd.Timestamp("2024-04-01")


def test_missing_input_error_lists_all_configured_files(tmp_path):
    config_path = tmp_path / "config" / "project.yaml"
    config_path.parent.mkdir()
    config = {
        "paths": {
            "profile": "data/profile.csv",
            "activity": "data/activity.csv",
            "labels": "data/labels.csv",
        }
    }
    with pytest.raises(FileNotFoundError) as error:
        input_table_paths(config, config_path)
    message = str(error.value)
    assert "profile.csv" in message
    assert "activity.csv" in message
    assert "labels.csv" in message
    assert "update their paths in config/project.yaml" in message
