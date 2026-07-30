from typing import Optional
import numpy as np
import pandas as pd
import heapq
from src.config import num_profile_points, split_feature, profile_features
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

    working_df = df.copy()
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


def process_simulation(sim):
    """ This function processes a simulation into 
        a list of profiles.

        Parameter(s):
            sim - a loaded numpy structured array which has all of the zones
                  of a simulation for a spatio-temporal evolution as rows

        Return Value(s):
            ages - a list of ages corresponding to each profile
            profiles - the profiles
    """
    zone_starts = np.where(sim["zone"] == 1.0)[0] # finding all zones at interior or exterior of star (depends on setup)

    profiles = []
    ages = []

    for i, start in enumerate(zone_starts):
        if i < len(zone_starts) - 1:
            profiles.append(
                sim.iloc[start:zone_starts[i + 1]] # getting zones belonging to profile
            )

        else:
            profiles.append(
                sim.iloc[start:] # edge case for last zone
            )

        ages.append(
            sim.iloc[start]["star_age"] # collecting ages
        )

    ages = np.array(ages)

    # ordering profiles by age
    ordered_inds = ages.argsort()
    profiles = np.array(profiles, dtype = object)
    profiles = profiles[ordered_inds]
    ages = ages[ordered_inds]

    return ages, profiles


# Return a simplified DataFrame with rows selected by the Ramer-Douglas-Peucker algorithm.
def get_max_perpendicular_distance(points, start_idx, end_idx):
    """
    Computes the maximum perpendicular distance from points between start_idx and end_idx 
    to the line segment connecting points[start_idx] and points[end_idx].
    Returns the maximum distance and the relative index of the point.
    """
    if end_idx - start_idx <= 1:
        return 0.0, -1
        
    A = points[start_idx]
    B = points[end_idx]
    
    # Vectors from A to B, and A to all intermediate points P
    AB = B - A
    P = points[start_idx + 1 : end_idx]
    AP = P - A
    
    AB_squared = np.dot(AB, AB)
    
    if AB_squared == 0:
        # Start and end points are identical; distance is just magnitude of AP
        distances = np.linalg.norm(AP, axis=1)
    else:
        # Project AP onto AB, clamp to [0, 1] to restrict to line segment
        t = np.dot(AP, AB) / AB_squared
        t = np.clip(t, 0.0, 1.0)
        
        # Calculate perpendicular distances (N-dimensional)
        projections = A + np.outer(t, AB)
        distances = np.linalg.norm(P - projections, axis=1)
        
    max_relative_idx = np.argmax(distances)
    max_dist = distances[max_relative_idx]
    split_idx = start_idx + 1 + max_relative_idx
    
    return max_dist, split_idx

def iterative_rdp_max_heap(points, original_indices, target_num_points):
    """
    Iterative Ramer-Douglas-Peucker algorithm utilizing a max heap.
    Prioritizes adding points that have the largest perpendicular distance 
    until target_num_points is reached.
    """
    n = len(points)
    if n <= target_num_points:
        return original_indices

    # Track selected points (using relative indices)
    selected_indices = {0, n - 1}
    
    # Heap stores: (-distance, start_idx, end_idx, split_idx)
    # Negative distance forces heapq (a min-heap) to act as a max-heap.
    heap = []
    
    dist, split_idx = get_max_perpendicular_distance(points, 0, n - 1)
    if split_idx != -1:
        heapq.heappush(heap, (-dist, 0, n - 1, split_idx))
        
    while len(selected_indices) < target_num_points and heap:
        neg_dist, start_idx, end_idx, split_idx = heapq.heappop(heap)
        
        selected_indices.add(split_idx)
        
        # Evaluate left segment
        left_dist, left_split = get_max_perpendicular_distance(points, start_idx, split_idx)
        if left_split != -1:
            heapq.heappush(heap, (-left_dist, start_idx, split_idx, left_split))
            
        # Evaluate right segment
        right_dist, right_split = get_max_perpendicular_distance(points, split_idx, end_idx)
        if right_split != -1:
            heapq.heappush(heap, (-right_dist, split_idx, end_idx, right_split))
            
    # Sort to maintain the temporal/sequential order of the profile
    sorted_selected = sorted(list(selected_indices))
    return original_indices[sorted_selected]

def rdp_preprocess(raw_df, num_profile_points=num_profile_points, split_feature=split_feature, profile_features=profile_features):
    # Normalized minmax scaled (between 0 & 1)
    normalized_df, _ = fit_preprocess_scalers(raw_df, profile_features + [split_feature], True, False)    
    # Stratify data into profiles based on a split feature
    # Expected output: lists of DataFrames (profiles) retaining original indices, and their split values
    splitting_features, profiles = process_simulation(normalized_df)

    # Determine which profile has the minimum number of points
    min_points = min([len(p) for p in profiles])

    # Ensure inputted num_profile_points is acceptable 
    if num_profile_points > min_points:
        num_profile_points = min_points

    selected_absolute_indices = []

    for profile, feature_val in zip(profiles, splitting_features):
        # Extract N-dimensional points and original dataframe indices
        points = profile[profile_features].to_numpy()
        original_indices = profile.index.to_numpy()
        
        # Implement RDP algorithm (iterative max-heap)
        rdp_indices = iterative_rdp_max_heap(points, original_indices, num_profile_points)
        selected_absolute_indices.extend(rdp_indices)
        
    # Create numpy array containing data from raw_df
    # Select specific columns and the indices identified by RDP
    final_columns = profile_features + [split_feature]
    filtered_df = raw_df.loc[selected_absolute_indices, final_columns]
    
    # Return as numpy array
    return filtered_df.to_numpy()