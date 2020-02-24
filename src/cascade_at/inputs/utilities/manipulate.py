import numpy as np

from cascade_at.dismod.api.fill_extract_helpers.utils import array_start_mid_end


def get_midpoint_dict(observed, grid_array):
    """
    Associate the midpoint of a grid_array with tuples of the observed
    dictionary of lower and upper times or ages. To be used in mapping
    the tuples to the midpoint.

    Args:
        observed: np.ndarray with shape (x, 2) where
            [:,0] is the lower value
            [:,1] is the upper value
        grid_array: np.ndarray with shape (y, 3) where
            [:,0] is the lower grid point
            [:,1] is the midpoint
            [:,2] is the upper grid point

    Returns:
            dictionary keyed by tuples mapping to a float
    """
    dictionary = {}

    for i in range(grid_array.shape[1]):
        grab = ((observed[:, 1] > grid_array[0, i]) &
                (observed[:, 0] < grid_array[2, i]))
        bucket = list(map(tuple, observed[grab]))
        for j in bucket:
            if j in dictionary.keys():
                raise RuntimeError("Cannot have a finer grid than observed data.")
            dictionary[j] = grid_array[1, i]
    return dictionary


def get_midpoint_grid_from_data(df, grid):
    """
    Averages meas_value and meas_std over (time_lower, time_upper)
    and (age_lower, age_upper) values that fall within the grid points.

    IMPORTANT: Grid cannot be finer than the actual data
    TODO: Put in a check for this and throw error

    Args:
        df: pd.DataFrame with columns 'location_id', 'sex_id', 'age_lower',
            'age_upper', 'time_lower', 'time_upper', 'meas_value', 'meas_std'
        grid: dict of np.array with keys for 'age' and 'time'

    Returns:
        pd.DataFrame with same columns as input but (potentially) reduced dimension
        length and age_lower = age_upper, time_lower = time_upper,
        averaged meas_value and meas_std.
    """

    data = df.copy()

    age_arr = array_start_mid_end(grid['age'])
    time_arr = array_start_mid_end(grid['time'])

    ages = np.asarray(data[['age_lower', 'age_upper']].drop_duplicates())
    times = np.asarray(data[['time_lower', 'time_upper']].drop_duplicates())

    age_dict = get_midpoint_dict(observed=ages, grid_array=age_arr)
    time_dict = get_midpoint_dict(observed=times, grid_array=time_arr)

    data['age_tup'] = list(zip(data.age_lower, data.age_upper))
    data['time_tup'] = list(zip(data.time_lower, data.time_upper))

    data['age_mid'] = data['age_tup'].map(age_dict)
    data['time_mid'] = data['time_tup'].map(time_dict)

    data = data.loc[~data.time_mid.isnull()]
    data = data.loc[~data.age_mid.isnull()]

    data = data[['location_id', 'sex_id', 'age_mid', 'time_mid',
                 'meas_value', 'meas_std']]
    data = data.groupby(['location_id', 'sex_id', 'age_mid', 'time_mid']).mean()
    data = data.reset_index()

    data.rename(columns={'time_mid': 'time_lower',
                         'age_mid': 'age_lower'}, inplace=True)
    data['time_upper'] = data['time_lower']
    data['age_upper'] = data['age_lower']

    data = data[['location_id', 'sex_id', 'age_lower', 'age_upper',
                 'time_lower', 'time_upper', 'meas_value', 'meas_std']]

    return data
