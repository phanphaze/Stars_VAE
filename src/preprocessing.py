from __future__ import annotations
from typing import Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Return a summary of missing and null values in a DataFrame.
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.isna()
        .sum()
        .to_frame(name="missing_values")
        .sort_values("missing_values", ascending=False)
    )
    summary["null_percent"] = (summary["missing_values"] / len(df) * 100).round(2)
    return summary[summary["missing_values"] > 0]

# Drop columns with constant values from the DataFrame.
def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_columns = df.columns[df.nunique() <= 1]
    if len(constant_columns) > 0:
        print(f"Dropping constant columns: {list(constant_columns)}")
        df = df.drop(columns=constant_columns)
    return df

# Return a list of numeric feature names from the DataFrame.
def numeric_features(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()

# fits the data to the scalers and returns the transformed DataFrame and the fitted scalers
def fit_preprocess_scalers(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    normalize: bool = True,
    standardize: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if features is None:
        features = numeric_features(df)

    working_df = drop_constant_columns(df.copy())
    scalers: dict[str, object] = {}

    for feature in features:
        if feature not in working_df.columns:
            continue

        values = pd.to_numeric(working_df[feature], errors="coerce").astype(float)
        values = values.fillna(values.median())

        if values.nunique(dropna=True) <= 1:
            working_df[feature] = 0.0
            continue

        transformer = None
        if normalize:
            transformer = MinMaxScaler()
        elif standardize:
            transformer = StandardScaler()

        if transformer is not None:
            working_df[[feature]] = transformer.fit_transform(values.to_frame())
            scalers[feature] = transformer

    return working_df, scalers

# transform the data using the provided scalers and return the transformed DataFrame
def transform_with_scalers(
    df: pd.DataFrame,
    scalers: dict[str, object],
) -> pd.DataFrame:
    working_df = df.copy()
    for feature, scaler in scalers.items():
        if feature not in working_df.columns:
            continue
        values = pd.to_numeric(working_df[feature], errors="coerce").astype(float)
        values = values.fillna(values.median())
        working_df[[feature]] = scaler.transform(values.to_frame())
    return working_df

# Return a simplified DataFrame with rows selected by the Ramer-Douglas-Peucker algorithm.
def rdp(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    if features is None:
        features = numeric_features(df)

    working_df = drop_constant_columns(df.copy())
    selected_features = [feature for feature in features if feature in working_df.columns]
    if not selected_features or len(working_df) < 3:
        return working_df

    points = (
        working_df[selected_features]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .fillna(lambda col: col.median())
    )
    points = points.fillna(points.median(axis=0))

    indices = _rdp_indices(points.to_numpy(), epsilon)
    simplified_df = working_df.iloc[indices].reset_index(drop=True)
    return simplified_df