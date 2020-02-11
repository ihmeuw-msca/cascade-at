import pytest
import numpy as np

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.model.utilities.grid_helpers import integrand_grids


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
def alchemy():
    return Alchemy(load_settings(BASE_CASE))


def test_integrand_grids(alchemy, default_ages, default_times):
    g = integrand_grids(alchemy=alchemy, integrands=['iota', 'prevalence'])
    np.testing.assert_array_equal(g['prevalence']['age'], default_ages)
    np.testing.assert_array_equal(g['prevalence']['time'], default_times)
    np.testing.assert_array_equal(g['iota']['age'], np.array([0., 5., 10., 50., 100.]))
    np.testing.assert_array_equal(g['iota']['time'], np.array([1990., 1995., 2000., 2005., 2010., 2015., 2016.]))
