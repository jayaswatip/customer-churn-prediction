"""Generate deterministic, fictional tables for exercising the demo pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 42
CUSTOMER_COUNT = 1600


def main() -> None:
    rng = np.random.default_rng(SEED)
    RAW.mkdir(parents=True, exist_ok=True)

    customer_ids = np.array([f"DEMO-{number:05d}" for number in range(1, CUSTOMER_COUNT + 1)])
    signup_start = pd.Timestamp("2022-01-01")
    signup_end = pd.Timestamp("2023-12-31")
    signup_days = rng.integers(0, (signup_end - signup_start).days + 1, CUSTOMER_COUNT)
    signup_dates = signup_start + pd.to_timedelta(signup_days, unit="D")
    age = rng.integers(18, 76, CUSTOMER_COUNT).astype(float)
    gender = rng.choice(["Female", "Male", "Non-binary"], CUSTOMER_COUNT, p=[0.49, 0.48, 0.03])
    cities = rng.choice(
        ["Northport", "Lakeview", "Riverton", "Hillcrest", "Westfield"],
        CUSTOMER_COUNT,
        p=[0.28, 0.22, 0.20, 0.17, 0.13],
    )
    tiers = rng.choice(["Bronze", "Silver", "Gold", "Platinum"], CUSTOMER_COUNT, p=[0.42, 0.32, 0.19, 0.07])
    risk = rng.normal(0.0, 1.0, CUSTOMER_COUNT)
    tier_value = pd.Series(tiers).map(
        {"Bronze": -0.15, "Silver": 0.0, "Gold": 0.20, "Platinum": 0.35}
    ).to_numpy()
    email_open_rate = np.clip(
        0.52 - 0.17 * risk + 0.08 * tier_value + rng.normal(0, 0.12, CUSTOMER_COUNT),
        0,
        1,
    )

    activity_rows: list[dict[str, object]] = []
    for customer_index, customer_id in enumerate(customer_ids):
        first_month = max(
            signup_dates[customer_index].to_period("M"),
            pd.Period("2023-01", freq="M"),
        )
        for month in pd.period_range(first_month, "2024-06", freq="M"):
            seasonal = 1.0 + 0.08 * np.sin(month.month * 2 * np.pi / 12)
            transactions = int(
                rng.poisson(max(0.45, 4.0 - 0.85 * risk[customer_index] + 1.8 * tier_value[customer_index]))
            )
            sessions = int(
                rng.poisson(max(0.4, 8.0 - 1.8 * risk[customer_index] + 2.2 * tier_value[customer_index]))
            )
            spend = max(
                0.0,
                float(
                    rng.gamma(
                        shape=max(1.0, transactions * 1.5),
                        scale=max(2.0, 13.0 + 2.5 * tier_value[customer_index]),
                    )
                    * seasonal
                ),
            )
            redeemed = int(rng.poisson(max(0.1, 0.7 + 0.4 * tier_value[customer_index])))
            tickets = int(rng.poisson(max(0.05, 0.22 + 0.17 * risk[customer_index])))
            activity_rows.append(
                {
                    "customer_id": customer_id,
                    "activity_month": month.to_timestamp().strftime("%Y-%m-%d"),
                    "monthly_spend": round(spend, 2),
                    "transaction_count": transactions,
                    "app_sessions": sessions,
                    "rewards_redeemed": redeemed,
                    "support_tickets": tickets,
                }
            )

    activity = pd.DataFrame(activity_rows)
    # Add a small number of conspicuous but documented demo quality issues so
    # the corresponding checks can be seen working.
    missing_month_customers = rng.choice(customer_ids, size=20, replace=False)
    activity = activity.loc[
        ~(
            activity["customer_id"].isin(missing_month_customers)
            & activity["activity_month"].eq("2023-11-01")
        )
    ].copy()
    negative_spend_index = activity.index[0]
    activity.loc[negative_spend_index, "monthly_spend"] = -12.0
    zero_transaction_indices = activity.index[1:11]
    activity.loc[zero_transaction_indices, "transaction_count"] = 0
    activity.loc[zero_transaction_indices, "monthly_spend"] = 0.0
    recent = activity.loc[
        activity["activity_month"].between("2023-10-01", "2024-03-01")
    ].groupby("customer_id")
    support_6m = recent["support_tickets"].sum().reindex(customer_ids, fill_value=0).to_numpy()
    recent_spend = recent["monthly_spend"].mean().reindex(customer_ids).fillna(0).to_numpy()
    # The label depends on pre-window behaviour, engagement and support plus
    # noise; it is fictional and intended only to exercise model training.
    log_odds = (
        -1.65
        + 0.88 * risk
        - 0.006 * (recent_spend - 160)
        - 0.20 * email_open_rate
        + 0.14 * support_6m
        - 0.16 * tier_value
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    churned = rng.binomial(1, churn_probability)
    labels = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "churn_status": np.where(churned == 1, "churned", "active"),
        }
    )
    profile = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates.strftime("%Y-%m-%d"),
            "age": age,
            "gender": gender,
            "membership_tier": tiers,
            "city": cities,
            "email_open_rate": email_open_rate.round(3),
            "support_tickets_6m": support_6m,
        }
    )
    profile.loc[rng.choice(profile.index, size=12, replace=False), "age"] = np.nan
    profile.loc[rng.choice(profile.index, size=15, replace=False), "email_open_rate"] = np.nan
    casing_indices = rng.choice(profile.index, size=24, replace=False)
    for index in casing_indices[:8]:
        profile.loc[index, "city"] = str(profile.loc[index, "city"]).lower()
    for index in casing_indices[8:16]:
        profile.loc[index, "city"] = str(profile.loc[index, "city"]).upper()

    profile.to_csv(RAW / "fb_customer_profile_demo.csv", index=False)
    activity.to_csv(RAW / "fb_monthly_activity_demo.csv", index=False)
    labels.to_csv(RAW / "fb_churn_label_demo.csv", index=False)
    print(f"Wrote {len(profile):,} fictional demo customers to {RAW}")
    print(f"Wrote {len(activity):,} pre- and post-window activity rows to {RAW}")
    print(f"Synthetic churn rate: {labels['churn_status'].eq('churned').mean():.1%}")
    print("These generated data and statistics are not FreshBasket business data or findings.")


if __name__ == "__main__":
    main()
