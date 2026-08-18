"""Across-seed aggregation for the predictive-eval tidy long format.

The outer seed loop in scripts/eval_predictive_baseline.py is the *only* replicate
axis: exactly one independently-seeded run per (grid value, seed) cell. Every
downstream summary therefore reduces to "mean +/- std over seeds, per grid value" --
a flat groupby, with no within-seed-replicate-then-across-seed hierarchy to unwind.
"""


def seed_mean_std(df, x_key, value_col='value', seed_col='seed', extra_group_cols=()):
    """Mean/std/count of `value_col` across the seed axis, per (*extra_group_cols, x_key).

    Args:
        df: Tidy scalar frame, already filtered to one metric (and usually one dgp).
        x_key: Sweep axis column, e.g. 'ensemble_size' or 'dropout_p'.
        value_col: Column holding the metric value.
        seed_col: Column holding the outer seed; used only for the invariant check.
        extra_group_cols: Additional grouping keys, e.g. ('method',).

    Returns:
        Flat frame with columns [*extra_group_cols, x_key, 'mean', 'std', 'n_seeds'],
        sorted by the grouping keys.

    Two details that matter:

    - `dropna=False` is mandatory. `dropout_p` is NaN on every non-MC_Dropout row and
      `ensemble_size` is None on the oracle rows; pandas' default would silently drop
      those groups rather than aggregate them.
    - `std` is pandas' default ddof=1 -- the sample std across the 5 seeds, which is
      what we want for reporting spread. It is NaN for a single-seed run, so callers
      drawing error bars should `.fillna(0.0)`.

    Raises:
        ValueError: if any cell holds more than one row per seed. Under this design
            that means a replicate loop survived the restructure, which would silently
            understate the across-seed spread rather than fail loudly.
    """
    group_cols = [*extra_group_cols, x_key]

    out = (df.groupby(group_cols, dropna=False)[value_col]
             .agg(mean='mean', std='std', n_seeds='count')
             .reset_index()
             .sort_values(group_cols)
             .reset_index(drop=True))

    if seed_col in df.columns:
        n_unique = (df.groupby(group_cols, dropna=False)[seed_col]
                      .nunique()
                      .reset_index(drop=True))
        if (out['n_seeds'].to_numpy() != n_unique.to_numpy()).any():
            raise ValueError(
                f"More than one row per ({group_cols}, {seed_col}) cell. This design "
                "expects exactly one independently-seeded run per (grid value, seed).")

    return out
