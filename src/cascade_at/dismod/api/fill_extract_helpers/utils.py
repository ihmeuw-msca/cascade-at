import pandas as pd
import numpy as np


def array_start_mid_end(array):
    """
    Computes the starting point, ending point, and
    midpoint of an array.

    Args:
        array: (np.array)

    Returns:
        (np.array)
    """
    return np.array([
        array[:-1],
        vec_to_midpoint(array),
        array[1:]
    ])


def vec_to_midpoint(array):
    """
    Computes the midpoint between elements in an array.
    Args:
        array: (np.array)
    Returns: (np.array)
    """
    return (array[1:] + array[:-1]) / 2


def map_locations_to_nodes(df, node_df):
    """
    Maps the location ID to node ID and
    changes column names in a df.
    """
    data = df.copy()
    data.rename(columns={
        "location_id": "c_location_id",
        "location": "c_location"
    }, inplace=True)
    data['c_location_id'] = data['c_location_id'].astype(int)
    data = data.merge(
        node_df[["node_id", "c_location_id"]],
        on=["c_location_id"]
    )
    return data


def map_covariate_names(df, covariate_df):
    """
    Maps the covariate names to the covariate
    IDs in the covariate table.
    """
    data = df.copy()
    covariate_rename = pd.Series(
        covariate_df.covariate_name.values,
        index=covariate_df.c_covariate_name
    ).to_dict()

    data.rename(columns=covariate_rename, inplace=True)
    return data


def convert_age_time_to_id(df, age_df, time_df):
    """
    Converts the times and ages to IDs based on a dictionary passed
    that should be made from the age or time table. Gets the "closest"
    age or time.

    :param df: pd.DataFrame
    :param age_df: pd.DataFrame
    :param time_df: pdDataFrame
    :return:
    """
    at_tables = {'age': age_df, 'time': time_df}
    assert "age" in df.columns
    assert "time" in df.columns
    df = df.assign(save_idx=df.index)
    for dat in ["age", "time"]:
        col_id = f"{dat}_id"
        sort_by = df.sort_values(dat)
        in_grid = sort_by[dat].notna()
        at_table = at_tables[dat]
        aged = pd.merge_asof(sort_by[in_grid], at_table, on=dat, direction="nearest")
        df = df.merge(aged[["save_idx", col_id]], on="save_idx", how="left")
    assert "age_id" in df.columns
    assert "time_id" in df.columns
    return df.drop(["save_idx", "age", "time"], axis=1)
