# Portfolio Data Quality

A runnable, synthetic demonstration of holdings reconciliation: detect defects, distinguish disagreement from evidence of error, rebuild a customer–symbol history for comparison, and record test outcomes.

This portfolio project demonstrates financial Product Owner work: defining what must agree across data routes, writing acceptance criteria, and turning exceptions into evidence engineers can reproduce. The schema, identifiers and fixtures belong to this educational implementation. They are not a production database specification, a deployed service, or proof that a real reconciliation passed.

## Run locally

Requires Python 3.10 or later. From this folder:

```bash
pip install -e ".[dev]"
pytest -q
portfolio-dq catalogue
portfolio-dq generate --out demo-data --defects all
portfolio-dq check --db demo-data --xlsx tracker.xlsx
portfolio-dq scenarios --xlsx scenario-tracker.xlsx
portfolio-dq compress --db demo-data
```

`check` reports injected defects without failing the shell by default. Add `--fail-on-exception` to use failures as a gate. `scenarios` returns a nonzero exit code for missed, unexpected, misclassified or failed comparisons. Generated workbooks are local demonstration outputs; the CLI does not call a brokerage service.

## How it works

Two independent routes describe the same holdings: a depository feed and a transaction ledger with corporate actions. Checks test consistency within each route and agreement between routes. A disagreement is an investigation candidate; neither side automatically becomes the truth.

| Part | Purpose |
|---|---|
| `src/portfolio_dq/checks/` | Check catalogue, scope, severity, denominator and evidence |
| `replay.py`, `pipeline.py` | Local rebuild-and-compare reference implementation |
| `scenarios.py` | Positive and negative controls, with explicit test outcomes |
| `tracker.py` | Excel exception and scenario registers |
| `compression.py` | Daily-to-range compression and reproducible fingerprints |
| `src/portfolio_dq/sql/` | Queries against this synthetic schema, not production SQL Server scripts |
| `generator.py` | Seeded synthetic data and an injected-defect manifest |

The replay is report-only. It does not repair a source system. Its reconstruction rules are toolkit assumptions and must not be mistaken for the financial methodology of any employer.

## Acceptance criteria

- A test identifies the customer–symbol, date, table, field and expected condition before comparison.
- Clean fixtures do not produce unexpected discrepancies.
- Broken fixtures produce the expected defect classes; informational differences remain distinct from errors.
- A successful command or HTTP response alone never proves discrepancy detection.
- Rebuild results are compared to stored evidence; differences are retained for review.
- Range compression preserves covered customer–symbol–days and stable fingerprints.

The automated suite in `tests/` checks these behaviours using synthetic fixtures. It does not establish production accuracy or a real UAT sign-off.

## Data and confidentiality

Recreate examples with `python scripts/make_example_data.py`. All bundled CSVs are demonstration data. Do not replace them with customer exports. The masking utilities are numerical transformations for experiments; they do **not** anonymize identities or make real records suitable for publication.

Dates, quantities, prices, fees and statuses follow the invented schema in `dataset.py` and `model.py`. Fee schedules and trading rules are illustrative. Before adapting the toolkit, explicitly agree currency, units, corporate-action dates, lineage, rounding and financial reconstruction with the responsible domain experts.

MIT license. Personal educational project; no employer endorsement or deployment claim.
