import pytest
import pandas as pd
import numpy as np

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.model.utilities.grid_helpers import integrand_grids
from cascade_at.inputs.utilities.manipulate import get_midpoint_grid_from_data, get_midpoint_dict


@pytest.fixture(scope='module')
def alchemy():
    return Alchemy(load_settings(BASE_CASE))


@pytest.fixture(scope='module')
def fake_csmr_data():
    num_years = len(range(2000, 2001))
    num_ages = len(np.array([0., 1.917808e-02, 7.671233e-02, 1.]))
    return pd.DataFrame({
        'location_id': np.repeat(101, num_years*num_ages),
        'sex_id': np.repeat(2, num_years*num_ages),
        'time_lower': np.repeat(list(range(2000, 2001)), num_ages),
        'time_upper': np.repeat(list(range(2001, 2002)), num_ages),
        'age_lower': np.tile(np.array([0., 1.917808e-02, 7.671233e-02, 1.]), num_years),
        'age_upper': np.tile(np.array([1.917808e-02, 7.671233e-02, 1.] + list(range(5, 10, 5))), num_years),
        'meas_value': np.random.uniform(0.01, 1, size=num_years*num_ages),
        'meas_std': np.random.uniform(0, 0.01, size=num_years*num_ages)
    })


@pytest.mark.parametrize("observed_ages,grid,answer", [
    (np.array([[0., 0.3], [0.3, 5]]), np.array([[0.1, 0.2, 0.5], [0.5, 1., 1.7]]).transpose(), {(0., 0.3): 0.2, (0.3, 5): 1.})
])
def test_get_midpoint_dict(observed_ages, grid, answer):
    np.testing.assert_equal(answer, get_midpoint_dict(observed_ages, grid))


def test_get_midpoint_grid_from_data(alchemy, fake_csmr_data):
    g = integrand_grids(alchemy=alchemy, integrands=['chi'])['chi']
    df = get_midpoint_grid_from_data(fake_csmr_data, g)
    assert len(df) == 4
    assert all(df.location_id == 101)
    assert all(df.sex_id == 2)
    # These exactly line up with the grid
    np.testing.assert_almost_equal(df.age_lower.values, np.array([0.009589, 0.047945205, 0.5383561649, 3.]))
    np.testing.assert_almost_equal(df.age_upper.values, np.array([0.009589, 0.047945205, 0.5383561649, 3.]))
    # It's 2002.5 because of the grid
    assert all(df.time_lower == 2002.5)
    assert all(df.time_upper == 2002.5)
