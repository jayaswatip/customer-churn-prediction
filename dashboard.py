from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@st.cache_data
def load_validation_results() -> pd.DataFrame:
    path = OUTPUTS / "validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = OUTPUTS / "customer_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_scenario() -> pd.DataFrame:
    path = OUTPUTS / "retention_scenario.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    path = OUTPUTS / "feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def render_metric_cards(validation: pd.DataFrame, predictions: pd.DataFrame, scenario: pd.DataFrame) -> None:
    churn_rate = predictions["churn"].mean() * 100
    best_model = validation.sort_values("pr_auc", ascending=False).iloc[0]
    scenario_row = scenario.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers scored", f"{len(predictions):,}")
    col2.metric("Observed churn rate", f"{churn_rate:.1f}%")
    col3.metric("Best PR-AUC", f"{best_model['pr_auc']:.3f}")
    col4.metric("Incremental retained", f"{scenario_row['estimated_incremental_retained_customers']:.1f}")


def render_model_table(validation: pd.DataFrame) -> None:
    st.subheader("Model comparison")
    display = validation[["model", "feature_set", "precision", "recall", "f1", "pr_auc", "accuracy"]].copy()
    display["pr_auc"] = display["pr_auc"].round(4)
    display["f1"] = display["f1"].round(4)
    display["accuracy"] = display["accuracy"].round(4)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_risk_distribution(predictions: pd.DataFrame) -> None:
    st.subheader("Risk score distribution")
    bins = pd.cut(predictions["churn_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    hist = pd.DataFrame({"risk_band": bins, "customers": 1}).groupby("risk_band").sum().reset_index()
    hist["risk_band"] = hist["risk_band"].astype(str)
    st.bar_chart(hist, x="risk_band", y="customers")


def render_top_customers(predictions: pd.DataFrame) -> None:
    st.subheader("Highest-risk customers")
    top = predictions.sort_values("churn_probability", ascending=False).head(10).copy()
    columns = ["customer_id", "churn", "signup_cohort", "churn_probability", "predicted_churn", "key_model_cues"]
    top["churn_probability"] = top["churn_probability"].round(4)
    st.dataframe(top[columns], use_container_width=True, hide_index=True)


def render_scenario_card(scenario: pd.DataFrame) -> None:
    st.subheader("Retention scenario")
    row = scenario.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers contacted", f"{int(row['customers_contacted']):,}")
    c2.metric("Contact rate", f"{row['contact_fraction_assumption'] * 100:.1f}%")
    c3.metric("Assumed lift", f"{row['assumed_absolute_retention_lift'] * 100:.1f}%")
    c4.metric("Net value", f"${row['estimated_net_value']:.2f}")
    st.caption(row["causal_claim"])


def render_feature_importance(feature_importance: pd.DataFrame) -> None:
    st.subheader("Feature importance (validation PR-AUC)")
    top_features = feature_importance.sort_values("permutation_importance", ascending=False).head(8)
    st.bar_chart(top_features.set_index("feature")["permutation_importance"])


def render_outputs_gallery() -> None:
    st.subheader("Generated charts")
    chart_names = [
        "churn_by_signup_cohort.png",
        "feature_importance.png",
        "label_distribution.png",
        "numeric_churn_differences.png",
    ]
    for name in chart_names:
        path = OUTPUTS / name
        if path.exists():
            st.image(str(path), caption=name)


def main() -> None:
    st.set_page_config(page_title="FreshBasket Churn Dashboard", page_icon="📊", layout="wide")
    st.title("FreshBasket Retail Loyalty Churn Dashboard")

    try:
        validation = load_validation_results()
        predictions = load_predictions()
        scenario = load_scenario()
        feature_importance = load_feature_importance()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run the pipeline first with `freshbasket-churn run` or `python scripts/generate_demo_data.py` if the synthetic inputs have not yet been created.")
        return

    render_metric_cards(validation, predictions, scenario)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model results", "Customers", "Reports"])

    with tab1:
        render_risk_distribution(predictions)
        render_scenario_card(scenario)

    with tab2:
        render_model_table(validation)
        render_feature_importance(feature_importance)

    with tab3:
        render_top_customers(predictions)

    with tab4:
        render_outputs_gallery()


if __name__ == "__main__":
    main()
