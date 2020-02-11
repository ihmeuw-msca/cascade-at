import pytest
import numpy as np
from copy import deepcopy

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.model.grid_alchemy import Alchemy


@pytest.fixture(scope='module')
def modified_settings():
    s = deepcopy(BASE_CASE)
    s['model']['constrain_omega'] = 0
    return load_settings(s)


@pytest.fixture(scope='module')
def default_ages():
    return np.array([
        0., 1.917808e-02, 7.671233e-02, 1., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.
    ])


@pytest.fixture(scope='module')
def default_times():
    return np.array([
        1990., 1995., 2000., 2005., 2010., 2015., 2016.
    ])


@pytest.fixture(scope='module')
def alchemy(modified_settings):
    return Alchemy(modified_settings)


def test_construct_age_time_grid(alchemy, default_ages, default_times):
    grid = alchemy.construct_age_time_grid()
    np.testing.assert_array_equal(grid['time'], default_times)
    np.testing.assert_array_equal(grid['age'], default_ages)


def test_construct_single_age_time_grid(alchemy):
    grid = alchemy.construct_single_age_time_grid()
    np.testing.assert_array_equal(grid[1], np.array([2005.]))
    np.testing.assert_array_equal(grid[0], np.array([0.]))


def test_get_smoothing_grid(modified_settings, alchemy):
    rate_settings = {c.rate: c for c in modified_settings.rate}
    sm = alchemy.get_smoothing_grid(rate=rate_settings['iota'])
    np.testing.assert_array_equal(sm.ages, np.array([0., 5., 10., 50., 100.]))
    np.testing.assert_array_equal(sm.times, np.array([1990., 1995., 2000., 2005., 2010., 2015., 2016.]))


def test_get_all_smooth_grids(alchemy, default_ages, default_times):
    all_grids = alchemy.get_all_rates_grids()
    np.testing.assert_array_equal(all_grids['iota'].ages, np.array([0., 5., 10., 50., 100.]))
    np.testing.assert_array_equal(all_grids['iota'].times, np.array([1990., 1995., 2000., 2005., 2010., 2015., 2016.]))
    np.testing.assert_array_equal(all_grids['chi'].ages, default_ages)
    np.testing.assert_array_equal(all_grids['chi'].times, default_times)
    np.testing.assert_array_equal(all_grids['pini'].ages, np.array([0.]))
    np.testing.assert_array_equal(all_grids['pini'].times, np.array([2005.]))


@pytest.fixture(scope='module')
def model(alchemy, mi):
    return alchemy.construct_two_level_model(
        location_dag=mi.location_dag,
        parent_location_id=70,
        covariate_specs=mi.covariate_specs
    )


def test_iota_grid(model):
    np.testing.assert_array_equal(model['rate']['iota'].ages, np.array([0., 5., 10., 50., 100.]))
    np.testing.assert_array_equal(model['rate']['iota'].times,
                                  np.array([1990., 1995., 2000., 2005., 2010., 2015., 2016.]))


def test_chi_grid(model, default_ages, default_times):
    np.testing.assert_array_equal(model['rate']['chi'].ages, default_ages)
    np.testing.assert_array_equal(model['rate']['chi'].times, default_times)


def test_pini_grid(model):
    np.testing.assert_array_equal(model['rate']['pini'].ages, np.array([0.]))
    np.testing.assert_array_equal(model['rate']['pini'].times, np.array([2005.]))

