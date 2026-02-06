"""Feature set definitions.

Shared helper to select a thesis-consistent subset of engineered features.

This is primarily to keep Track A baselines stable on sparse panels where
long rolling windows (e.g., 52-week baselines) produce extensive NaNs.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Sequence


FeatureSetName = Literal["full", "core"]


CORE_FEATURE_SET_V01: Sequence[str] = (
    "feat_week_sin",
    "feat_week_cos",
    "feat_quarter",
    "feat_is_monsoon",
    "feat_lat_norm",
    "feat_lon_norm",
    "feat_lat_lon_interact",
    "feat_cases_lag_1",
    "feat_cases_lag_2",
    "feat_cases_lag_4",
    "feat_cases_ma_4w",
    "feat_cases_growth_rate",
    "feat_cases_var_4w",
    "feat_temp_lag_1",
    "feat_temp_lag_2",
    "feat_rain_lag_1",
    "feat_rain_lag_2",
    "feat_temp_anomaly",
    "feat_rain_persist_4w",
    "feat_degree_days",
)


def select_feature_columns(
    all_columns: Iterable[str],
    feature_set: FeatureSetName = "full",
) -> List[str]:
    columns = list(all_columns)

    if feature_set == "full":
        return [c for c in columns if c.startswith("feat_")]

    if feature_set != "core":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    present = set(columns)
    selected = [c for c in CORE_FEATURE_SET_V01 if c in present]
    if not selected:
        selected = [c for c in columns if c.startswith("feat_")]

    return selected
