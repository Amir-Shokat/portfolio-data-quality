"""portfolio_dq: precision monitoring and rebuild-and-compare testing for holdings data.

Holdings per customer and symbol are produced by two routes (a depository route and a
transactional route). This package generates a synthetic copy of such data with labelled
defects, runs a catalogue of checks that keeps within-route defects apart from cross-route
disagreements, rebuilds positions from events to compare with what is stored, and logs the
results in an Excel tracker.
"""

from .checks import catalogue, run_all, summary
from .dataset import Dataset
from .generator import DEFECTS, make_dataset
from .replay import rebuild_compare
from .scenarios import SCENARIOS, run_scenarios

__version__ = "1.0.0"
__all__ = ["DEFECTS", "Dataset", "SCENARIOS", "catalogue", "make_dataset", "rebuild_compare",
           "run_all", "run_scenarios", "summary"]
