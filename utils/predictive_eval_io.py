"""
Tidy-CSV writer for the predictive-distribution evaluation script.

utils.results_save has no CSV writer (save_statistics and friends are all xlsx via
openpyxl) -- this module adds one, additively, mirroring results_save's directory-
resolution / module-level-global-override / sanitize_filename conventions rather than
inventing a new one.
"""

from pathlib import Path

import pandas as pd

from utils.results_save import sanitize_filename

# Module-level variable for the output directory (mirrors results_save.plots_dir/
# stats_dir/outputs_dir -- can be overridden by the calling script).
csv_dir = None


def _default_csv_dir() -> Path:
    current = Path.cwd()
    project_root = current.parent if current.name == 'Experiments' else current
    return project_root / "results" / "predictive_eval" / "csv"


def save_tidy_csv(df: pd.DataFrame, filename: str, subfolder: str = '') -> Path:
    """Save a tidy (long-format) DataFrame as CSV under csv_dir (or its default),
    following the same subfolder/sanitize_filename/mkdir idiom as
    utils.results_save.save_plot/save_statistics."""
    global csv_dir
    if csv_dir is None:
        csv_dir = _default_csv_dir()

    save_dir = csv_dir / subfolder if subfolder else csv_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / f"{sanitize_filename(filename)}.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved CSV: {filepath} ({len(df)} rows)")
    return filepath
