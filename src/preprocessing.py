from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing and null values in a DataFrame."""
    summary = (
        df.isna()
        .sum()
        .to_frame(name="missing_values")
        .sort_values("missing_values", ascending=False)
    )
    summary["null_percent"] = (summary["missing_values"] / len(df) * 100).round(2)
    return summary[summary["missing_values"] > 0]


def preprocess_features(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    normalize: bool = True,
    standardize: bool = False,
) -> pd.DataFrame:
    """Scale each selected feature independently.

    This applies scaling per column rather than across columns. For example,
    ``initial_mass`` and ``initial_z`` are scaled using their own range, so they
    are not compressed by larger-magnitude columns such as ``star_age``.

    The function supports either Min-Max normalization or z-score standardization,
    but the default is normalization only to keep the transformed values in a
    bounded and interpretable range.
    """
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.tolist()

    working_df = df.copy()

    for feature in features:
        if feature not in working_df.columns:
            continue

        values = pd.to_numeric(working_df[feature], errors="coerce").astype(float)

        if values.isna().any():
            values = values.fillna(values.median())

        if values.nunique(dropna=True) <= 1:
            working_df[feature] = 0.0
            continue

        transformed = values.to_frame(name=feature)

        if normalize:
            transformed = pd.DataFrame(
                MinMaxScaler().fit_transform(transformed),
                index=working_df.index,
                columns=[feature],
            )

        if standardize:
            transformed = pd.DataFrame(
                StandardScaler().fit_transform(transformed),
                index=working_df.index,
                columns=[feature],
            )

        working_df[feature] = transformed[feature].to_numpy()

    return working_df
