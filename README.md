# FreshBasket Retail Loyalty Churn Prediction

This repository is a reproducible project scaffold for the April–June 2024
FreshBasket churn-label window. The original FreshBasket tables were not
provided, so the repository includes **synthetic demo tables with made-up
customers** to demonstrate the pipeline end to end. Synthetic metrics,
predictions, and recommendations are not FreshBasket findings and must not be
used to make business decisions. Replace the demo files with the real tables
and map their observed schema before drawing any real conclusions.

## Project layout

```text
config/project.yaml                  Explicit source, schema, split, and scenario settings
data/raw/                            Synthetic demo files; replace with original tables for real analysis
data/processed/                      Generated cleaned customer-level modelling dataset
notebooks/01_eda.ipynb               EDA and data-quality notebook
outputs/                             Generated reports, model, metrics, and charts
src/freshbasket_churn/               Importable pipeline modules and CLI
tests/                               Focused checks for validation and scenarios
```

## Setup in VS Code (Windows)

Open this folder in VS Code, then in its PowerShell terminal:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Select `.venv` as the Python interpreter in VS Code (`Python: Select
Interpreter`). If PowerShell blocks activation, use the interpreter directly:
`.venv\Scripts\python.exe`.

## Run the synthetic demo

The configured demo inputs are:

- `data/raw/fb_customer_profile_demo.csv`
- `data/raw/fb_monthly_activity_demo.csv`
- `data/raw/fb_churn_label_demo.csv`

They are generated deterministically with seed 42. To regenerate them:

```powershell
python scripts/generate_demo_data.py
```

The generator includes a few deliberate demo-quality cases: missing months,
missing profile values, inconsistent city casing, a negative spend value, and
zero-transaction rows. These make the checks visible; they are not claims about
FreshBasket data.

Inspect the demo columns and sample rows:

```powershell
freshbasket-churn inspect
```

Then run the full demonstration pipeline:

```powershell
freshbasket-churn run
```

Open a local dashboard built from the generated outputs:

```powershell
streamlit run dashboard.py
```

Every result from these files reflects synthetic data only.

## Replace the demo with real source tables

Replace the demo files in `data/raw/` or update the paths in
`config/project.yaml`. The default real-table examples are:

- `data/raw/fb_customer_profile.xlsx`
- `data/raw/fb_monthly_activity.xlsx`
- `data/raw/fb_churn_label.xlsx`

CSV files are supported too. For differently named Excel workbooks, update the
paths; the Excel reader uses the first worksheet. Inspect the inputs:

```powershell
freshbasket-churn inspect
```

This prints each table's exact column names, data types, missing counts,
distinct counts, and sample rows. `inspect` does not require the schema mappings.

## Map the observed schema

Edit `config/project.yaml` using only the actual names and label values reported
by `inspect`:

1. Set `schema.customer_id.profile`, `.activity`, and `.labels` to the customer
   key in each table.
2. Set `schema.activity_month` and `schema.signup_date` to their actual columns.
3. Set `schema.target` and explicitly map the observed churn and non-churn values
   under `schema.label_values`. Use YAML types matching the source values (for
   example, numeric `1` rather than the string `"1"`).
4. Map usable profile columns to a category in `schema.profile_features`, for
   example `actual age column: demographics`, `actual tier column: membership`,
   `actual city column: city`, or actual contact/support fields to `engagement`
   or `support`.
5. Map pre-window monthly numeric columns in `schema.monthly_features`, each
   with an appropriate group and aggregation (`sum`, `mean`, `max`, `min`, or
   chronological `last`). Do not map the activity month, identifiers, churn
   outcomes, or fields derived after the label window.
6. Configure observed spend fields in `spend_columns`, a transaction-count field
   in `transaction_count_column`, casing-review fields in
   `categorical_columns`, and any target-like fields in
   `forbidden_feature_columns`.
   Also change `dataset_label` from its synthetic-demo warning to an accurate
   description of the real input data. The report, presentation, and prediction
   file carry this label to discourage accidental misrepresentation.
7. Leave the label dates as April 1–June 30, 2024 unless the source definition
   explicitly differs. Confirm that the churn table represents that window and
   contains one row per customer. This scaffold deliberately errors on duplicate
   profile/label customer IDs rather than silently aggregating labels.

The config's schema mapping is explicit by design: unknown columns are not
silently treated as safe model features. If the source label table has multiple
rows per customer or requires period filtering, inspect its real structure and
extend the label mapping logic before running; do not arbitrarily select a row.

## Run

```powershell
freshbasket-churn inspect
freshbasket-churn run
```

Equivalent module command if the console script isn't on PATH:

```powershell
python -m freshbasket_churn.cli inspect
python -m freshbasket_churn.cli run
```

The configured cohort validation reserves the latest distinct signup cohort(s)
as a temporal holdout. The pipeline requires both target classes in training and
validation; it stops with an actionable error instead of falling back to a
random split. Change `validation.holdout_latest_cohorts` only after reviewing
the observed cohort and class counts.

## Generated deliverables

After a successful run:

- `data/processed/cleaned_modeling_dataset.csv`: one row per labeled, matched
  customer with mapped features aggregated from activity strictly before
  April 1, 2024.
- `outputs/quality_*.csv`: missing values, duplicate keys, missing global and
  customer months, invalid configured spend, zero transactions, casing
  candidates, and numeric IQR outliers.
- `outputs/numeric_churn_relationships.csv` and
  `outputs/categorical_churn_relationships.csv`: descriptive label associations.
- `outputs/validation_results.csv`: Logistic Regression and class-balanced
  Random Forest results for demographics-only, behavioural/engagement/support,
  and all mapped features. Metrics include precision, recall, F1, PR-AUC, and
  accuracy.
- `outputs/trained_model.joblib`: selected all-features model fitted on the
  pre-holdout cohorts, along with feature and validation metadata.
- `outputs/customer_predictions.csv`: churn probabilities, 0.5-threshold
  labels, and globally important feature-value cues for **out-of-time validation
  customers only**. This avoids presenting training-set scores as independent
  customer predictions.
- `outputs/feature_importance.csv`: validation permutation importance scored
  by PR-AUC; it is predictive association, not causation.
- `outputs/retention_scenario.csv`: targeted highest-risk outreach simulation.
- `outputs/churn_by_signup_cohort.png`, `label_distribution.png`,
  `numeric_churn_differences.png`, and `feature_importance.png`.
- `outputs/final_report.md` and `outputs/retention_review_presentation.pptx`.

The presentation is structured for approximately 10–12 minutes, with speaker
notes. The EDA notebook is `notebooks/01_eda.ipynb`; select the project virtual
environment as its kernel.

## Retention scenario and limitations

The scenario contacts the configured fraction of the highest-risk **validation
customers** and multiplies that count by an explicitly configured assumed
absolute retention lift. Default lift, contact fraction, cost, and retained-value
inputs in `config/project.yaml` are placeholders—not estimates from FreshBasket
data. Update them using approved business assumptions. The result is a
what-if simulation, not demonstrated impact or a causal treatment effect. A
randomized outreach holdout (or another defensible causal design) is needed to
measure incremental retention. When running the synthetic demo, this scenario
is illustrative only and does not estimate FreshBasket impact.

This analysis is subject to label quality, selection into the observed
population, available signup cohorts, unobserved confounders, and the specific
feature mappings. IQR outliers and month gaps are flagged for investigation,
not removed automatically. Review generated reports and data governance
requirements before operational use.

## Tests

```powershell
python -m pytest
```
#   c u s t o m e r - c h u r n - p r e d i c t i o n  
 #   c u s t o m e r - c h u r n - p r e d i c t i o n  
 #   c u s t o m e r - c h u r n - p r e d i c t i o n - p r o j e c t  
 