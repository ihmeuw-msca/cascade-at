import pytest
import numpy as np
from numpy import nan, inf
import pandas as pd

from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.dismod.api.dismod_filler import DismodFiller


@pytest.fixture(scope='module')
def df(mi, settings, temp_directory):
    alchemy = Alchemy(settings)
    d = DismodFiller(
        path=temp_directory / 'temp.db',
        settings_configuration=settings,
        measurement_inputs=mi,
        grid_alchemy=alchemy,
        parent_location_id=70,
        sex_id=2
    )
    d.fill_for_parent_child()
    return d


@pytest.fixture(scope='module')
def value_prior(df):
    return df.prior.merge(df.smooth_grid, left_on='prior_id', right_on='value_prior_id').copy()


@pytest.fixture(scope='module')
def dage_prior(df):
    return df.prior.merge(df.smooth_grid, left_on='prior_id', right_on='dage_prior_id').copy()


@pytest.fixture(scope='module')
def dtime_prior(df):
    return df.prior.merge(df.smooth_grid, left_on='prior_id', right_on='dtime_prior_id').copy()


""" priors for iota """


@pytest.mark.parametrize("column,value", [
    ("lower", 1e-06),
    ("upper", 1e-02),
    ("mean", 1.5e-4),
    ("std", 1.5),
    ("eta", nan),
    ("nu", nan)
])
def test_iota_value_prior(value_prior, column, value):
    iota_prior = value_prior.loc[value_prior.prior_name.str.contains("^iota_*[0-9]", regex=True)]
    if np.isnan(value):
        assert iota_prior[column].isnull().all()
    else:
        assert all(iota_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0.0),
    ("std", 1e-02),
    ("eta", nan),
    ("nu", nan)
])
def test_iota_dage_prior(dage_prior, column, value):
    iota_prior = dage_prior.loc[dage_prior.prior_name.str.contains("^iota_*[0-9]", regex=True)]
    if np.isnan(value):
        assert iota_prior[column].isnull().all()
    else:
        assert all(iota_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0.0),
    ("std", 1e-02),
    ("eta", nan),
    ("nu", nan)
])
def test_iota_dtime_prior(dtime_prior, column, value):
    iota_prior = dtime_prior.loc[dtime_prior.prior_name.str.contains("^iota_*[0-9]", regex=True)]
    if np.isnan(value):
        assert iota_prior[column].isnull().all()
    else:
        assert all(iota_prior[column] == value)


""" priors for omega (includes constraint) """


@pytest.mark.parametrize("column,value", [
    ("lower", 0.169995),
    ("upper", 0.169995),
    ("mean", 0.169995),
    ("std", nan),
    ("eta", nan),
    ("nu", nan)
])
def test_omega_value_prior(value_prior, column, value):
    omega_prior = value_prior.loc[value_prior.prior_name.str.contains("^omega_*[0-9]", regex=True)]
    if np.isnan(value):
        assert omega_prior[column].isnull().all()
    else:
        assert all(omega_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -inf),
    ("upper", inf),
    ("mean", 0),
    ("std", nan),
    ("eta", nan),
    ("nu", nan)
])
def test_omega_dage_prior(dage_prior, column, value):
    omega_prior = dage_prior.loc[dage_prior.prior_name.str.contains("^omega_*[0-9]", regex=True)]
    if np.isnan(value):
        assert omega_prior[column].isnull().all()
    else:
        assert all(omega_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -inf),
    ("upper", inf),
    ("mean", 0),
    ("std", nan),
    ("eta", nan),
    ("nu", nan)
])
def test_omega_dtime_prior(dtime_prior, column, value):
    omega_prior = dtime_prior.loc[dtime_prior.prior_name.str.contains("^omega_*[0-9]", regex=True)]
    if np.isnan(value):
        assert omega_prior[column].isnull().all()
    else:
        assert all(omega_prior[column] == value)


""" priors for chi """


@pytest.mark.parametrize("column,value", [
    ("lower", 1e-6),
    ("upper", 1e-2),
    ("mean", 4e-4),
    ("std", 0.2),
    ("eta", nan),
    ("nu", nan)
])
def test_chi_value_prior(value_prior, column, value):
    chi_prior = value_prior.loc[value_prior.prior_name.str.contains("^chi_*[0-9]", regex=True)]
    if np.isnan(value):
        assert chi_prior[column].isnull().all()
    else:
        assert all(chi_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0),
    ("std", 1e-2),
    ("eta", nan),
    ("nu", nan)
])
def test_chi_dage_prior(dage_prior, column, value):
    chi_prior = dage_prior.loc[dage_prior.prior_name.str.contains("^chi_*[0-9]", regex=True)]
    if np.isnan(value):
        assert chi_prior[column].isnull().all()
    else:
        assert all(chi_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0),
    ("std", 1e-2),
    ("eta", nan),
    ("nu", nan)
])
def test_chi_dtime_prior(dtime_prior, column, value):
    chi_prior = dtime_prior.loc[dtime_prior.prior_name.str.contains("^chi_*[0-9]", regex=True)]
    if np.isnan(value):
        assert chi_prior[column].isnull().all()
    else:
        assert all(chi_prior[column] == value)


""" priors for pini (just one for each since no age time grid) """


@pytest.mark.parametrize("column,value", [
    ("lower", 0.0),
    ("upper", 0.2),
    ("mean", 0.1),
    ("std", 1.0),
    ("eta", 1e-6),
    ("nu", nan)
])
def test_pini_value_prior(value_prior, column, value):
    pini_prior = value_prior.loc[value_prior.prior_name.str.contains("^pini_*[0-9]", regex=True)]
    if np.isnan(value):
        assert pini_prior[column].isnull().all()
    else:
        assert all(pini_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0.0),
    ("std", nan),
    ("eta", nan),
    ("nu", nan)
])
def test_pini_dage_prior(dage_prior, column, value):
    pini_prior = dage_prior.loc[dage_prior.prior_name.str.contains("^pini_*[0-9]", regex=True)]
    if np.isnan(value):
        assert pini_prior[column].isnull().all()
    else:
        assert all(pini_prior[column] == value)


@pytest.mark.parametrize("column,value", [
    ("lower", -1.0),
    ("upper", 1.0),
    ("mean", 0.0),
    ("std", nan),
    ("eta", nan),
    ("nu", nan)
])
def test_pini_dtime_prior(dtime_prior, column, value):
    pini_prior = dtime_prior.loc[dtime_prior.prior_name.str.contains("^pini_*[0-9]", regex=True)]
    if np.isnan(value):
        assert pini_prior[column].isnull().all()
    else:
        assert all(pini_prior[column] == value)


""" iota random effect """


def test_iota_re(value_prior, dage_prior, dtime_prior):
    np.testing.assert_array_equal(value_prior.loc[value_prior.prior_name.str.startswith('iota_re')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([[-inf, inf, 0.0, 1.0, nan, nan]]))
    np.testing.assert_array_equal(dage_prior.loc[dage_prior.prior_name.str.startswith('iota_re')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([[-inf, inf, 0.0, nan, nan, nan]]))
    np.testing.assert_array_equal(dtime_prior.loc[dtime_prior.prior_name.str.startswith('iota_re')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([[-inf, inf, 0.0, nan, nan, nan]]))


""" alpha covariate multipliers """


def test_iota_alpha(value_prior, dage_prior, dtime_prior):
    np.testing.assert_array_equal(value_prior.loc[value_prior.prior_name.str.startswith('alpha_iota')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([
        [-1., 1., 0., nan, nan, nan],
        [-1., 1., 0., nan, nan, nan]
    ]))
    np.testing.assert_array_equal(dage_prior.loc[dage_prior.prior_name.str.startswith('alpha_iota')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([
        [-1., 1., 0., nan, nan, nan],
        [-inf, inf, 0., nan, nan, nan]
    ]))
    np.testing.assert_array_equal(dtime_prior.loc[dtime_prior.prior_name.str.startswith('alpha_iota')][[
        'lower', 'upper', 'mean', 'std', 'eta', 'nu'
    ]].values, np.array([
        [-1., 1., 0., nan, nan, nan],
        [-inf, inf, 0., nan, nan, nan]
    ]))


@pytest.fixture
def data():
    return pd.DataFrame.from_dict(
        {'data_id': {0: 0, 1: 1, 2: 2}, 'data_name': {0: '0', 1: '1', 2: '2'}, 'integrand_id': {0: 8, 1: 10, 2: 9},
         'density_id': {0: 4, 1: 4, 2: 4}, 'node_id': {0: 67, 1: 67, 2: 67}, 'weight_id': {0: 3, 1: 3, 2: 3},
         'subgroup_id': {0: 0, 1: 0, 2: 0},
         'hold_out': {0: 0, 1: 1, 2: 0}, 'meas_value': {0: 4e-05, 1: 0.17, 2: 5e-06},
         'meas_std': {0: 3e-06, 1: 0.010204269138493082, 2: 1.020426913849308e-06},
         'eta': {0: 1e-05, 1: 1e-05, 2: 1e-05}, 'nu': {0: nan, 1: nan, 2: nan}, 'age_lower': {0: 0.0, 1: 0.0, 2: 0.0},
         'age_upper': {0: 0.01917808, 1: 0.01917808, 2: 0.01917808}, 'time_lower': {0: 1990.0, 1: 1990.5, 2: 1990.5},
         'time_upper': {0: 1991.0, 1: 1990.5, 2: 1990.5}, 'x_0': {0: 0.96, 1: 0.96, 2: 0.96},
         'x_1': {0: 1.0, 1: 1.0, 2: 1.0}, 'x_2': {0: -0.5, 1: -0.5, 2: -0.5}}
    )


@pytest.fixture
def covariate():
    return pd.DataFrame.from_dict(
        {'covariate_id': {0: 0, 1: 1, 2: 2}, 'covariate_name': {0: 'x_0', 1: 'x_1', 2: 'x_2'},
         'reference': {0: 0.96, 1: 1.0, 2: -0.5},
         'max_difference': {0: 1e-10, 1: nan, 2: 0.5000000001},
         'c_covariate_name': {0: 'c_diabetes_fpg', 1: 's_one', 2: 's_sex'}}
    )


@pytest.fixture
def density():
    return pd.DataFrame.from_dict(
        {'density_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
         'density_name': {0: 'uniform', 1: 'gaussian', 2: 'laplace', 3: 'students', 4: 'log_gaussian', 5: 'log_laplace',
                          6: 'log_students'}}

    )


@pytest.fixture
def age():
    return pd.DataFrame.from_dict(
        {'age_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14},
         'age': {0: 0.0, 1: 0.01917808, 2: 0.07671233, 3: 1.0, 4: 5.0, 5: 10.0, 6: 20.0, 7: 30.0, 8: 40.0, 9: 50.0,
                 10: 60.0, 11: 70.0, 12: 80.0, 13: 90.0, 14: 100.0}}
    )


@pytest.fixture
def time():
    return pd.DataFrame.from_dict(
        {'time_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
         'time': {0: 1990.0, 1: 1995.0, 2: 2000.0, 3: 2005.0, 4: 2010.0, 5: 2015.0, 6: 2016.0}}
    )


@pytest.fixture
def integrand():
    return pd.DataFrame.from_dict(
        {'integrand_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12},
         'integrand_name': {0: 'Sincidence', 1: 'remission', 2: 'mtexcess', 3: 'mtother', 4: 'mtwith', 5: 'susceptible',
                            6: 'withC', 7: 'prevalence', 8: 'Tincidence', 9: 'mtspecific', 10: 'mtall',
                            11: 'mtstandard', 12: 'relrisk'},
         'minimum_meas_cv': {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1,
                             11: 0.1, 12: 0.1}}
    )


@pytest.fixture
def mulcov():
    return pd.DataFrame.from_dict(
        {'mulcov_id': {0: 0, 1: 1}, 'mulcov_type': {0: 'rate_value', 1: 'rate_value'}, 'rate_id': {0: 1, 1: 1},
         'integrand_id': {0: None, 1: None}, 'covariate_id': {0: 2, 1: 0}, 'group_smooth_id': {0: 5, 1: 6},
         'group_id': {0: 0, 1: 0}, 'subgroup_smooth_id': {0: None, 1: None}}
    )


@pytest.fixture
def option():
    return pd.DataFrame.from_dict(
        {'option_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
                       14: 14, 15: 15, 16: 16},
         'option_name': {0: 'parent_node_id', 1: 'random_seed', 2: 'ode_step_size', 3: 'rate_case',
                         4: 'meas_noise_effect', 5: 'max_num_iter_fixed', 6: 'print_level_fixed',
                         7: 'accept_after_max_steps_fixed', 8: 'tolerance_fixed', 9: 'max_num_iter_random',
                         10: 'print_level_random', 11: 'accept_after_max_steps_random', 12: 'tolerance_random',
                         13: 'age_avg_split', 14: 'quasi_fixed', 15: 'bound_frac_fixed', 16: 'zero_sum_child_rate'},
         'option_value': {0: '67', 1: '495279142', 2: '5.0', 3: 'iota_pos_rho_zero', 4: 'add_var_scale_log', 5: '200',
                          6: '5', 7: '5', 8: '1e-06', 9: '100', 10: '0', 11: '5', 12: '1e-06',
                          13: '0.01917808 0.07671233 1.0', 14: 'false', 15: '1e-08', 16: 'iota'}}
    )


def test_age(df, age):
    pd.testing.assert_frame_equal(df.age, age)


def test_time(df, time):
    pd.testing.assert_frame_equal(df.time, time)


def test_integrand(df, integrand):
    pd.testing.assert_frame_equal(df.integrand, integrand)


def test_density(df, density):
    pd.testing.assert_frame_equal(df.density, density)


def test_covariate(df, covariate):
    pd.testing.assert_frame_equal(df.covariate, covariate)


def test_data(df, data):
    pd.testing.assert_frame_equal(df.data, data)


def test_mulcov(df, mulcov):
    pd.testing.assert_frame_equal(df.mulcov, mulcov)


def test_option(df, option):
    pd.testing.assert_frame_equal(df.option, option)
