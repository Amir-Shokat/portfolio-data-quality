"""Regenerate the synthetic example data in examples/ (no real customer or company data).

    python scripts/make_example_data.py

Writes examples/data/*.csv: every table of the synthetic schema with all labelled defects
injected (fixed seed), plus injected_defects.csv, the manifest of what was broken where.
"""

from pathlib import Path

from portfolio_dq.generator import make_dataset

out = Path(__file__).resolve().parents[1] / "examples" / "data"
ds = make_dataset(seed=7, defects="all")
ds.to_csv(out)
print(f"wrote {len(ds.tables) + 1} files to {out}")
