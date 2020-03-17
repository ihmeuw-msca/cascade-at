import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace

import cascade_at.core.db
from cascade_at.inputs.data import CrosswalkVersion
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.inputs.csmr import CSMR
from cascade_at.inputs.asdr import ASDR
from cascade_at.inputs.covariate_data import CovariateData
from cascade_at.context.model_context import Context
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.inputs.population import Population
from cascade_at.inputs.locations import LocationDAG

cascade_at.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True


def pytest_addoption(parser):
    group = parser.getgroup("cascade")
    group.addoption("--ihme", action="store_true",
                    help="run functions requiring access to central comp and Dismod-AT")
    group.addoption("--signals", action="store_true",
                    help="tests using Unix signals can crash the Mac.")
    group.addoption("--dismod", action="store_true",
                    help="requires access to Dismod-AT command line")
    group.addoption("--cluster", action="store_true",
                    help="run functions requiring access to fair cluster")


@pytest.fixture(scope='session')
def ihme(request):
    return IhmeDbFuncArg(request)


class IhmeDbFuncArg:
    """
    Uses a pattern from https://pytest.readthedocs.io/en/2.0.3/example/attic.html
    """
    def __init__(self, request):
        if not request.config.getoption("ihme"):
            pytest.skip(f"specify --ihme to run tests requiring Central Comp databases")

        cascade_at.core.db.BLOCK_SHARED_FUNCTION_ACCESS = False


@pytest.fixture
def cluster(request):
    return ClusterFuncArg(request)


class ClusterFuncArg:
    """
    Uses a pattern from https://pytest.readthedocs.io/en/2.0.3/example/attic.html
    """
    def __init__(self, request):
        if not request.config.getoption("cluster"):
            pytest.skip(f"specify --cluster to run tests requiring the cluster")


@pytest.fixture
def dismod(request):
    return DismodFuncArg(request)


class DismodFuncArg:
    """Must be able to run dmdismod."""
    def __init__(self, request):
        if not request.config.getoption("dismod"):
            pytest.skip("specify --dismod to run tests requiring Dismod")


@pytest.fixture(scope="session")
def temp_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope='session')
def Demographics():
    d = SimpleNamespace()
    d.age_group_id = [2]
    d.location_id = [70]
    d.sex_id = [2]
    d.year_id = [1990]
    return d


@pytest.fixture(scope='session')
def dag(ihme):
    d = LocationDAG(location_set_version_id=544, gbd_round_id=6)
    return d


@pytest.fixture(scope='session')
def cv(Demographics, ihme):
    cv = CrosswalkVersion(crosswalk_version_id=1, exclude_outliers=1,
                          demographics=Demographics, conn_def='dismod-at-dev',
                          gbd_round_id=6)
    cv.raw = pd.DataFrame({
        'underlying_nid': np.nan,
        'nid': 230075,
        'field_citation_value': '',
        'source_type': '',
        'location_name': 'Canada',
        'location_id': 70,
        'sex': 'Female',
        'year_start': 1990,
        'year_end': 1990,
        'age_start': 0.0,
        'age_end': 0.01917808,
        'measure': 'incidence',
        'mean': 4e-05,
        'lower': 5e-05,
        'upper': 3e-05,
        'standard_error': 3e-06,
        'cases': 100.,
        'sample_size': 2000000.,
        'unit_type': '',
        'unit_value_as_published': 1,
        'uncertainty_type_value': 95,
        'representative_name': '',
        'urbanicity_type': '',
        'recall_type': '',
        'recall_type_value': np.nan,
        'sampling_type': '',
        'group': np.nan,
        'specificity': np.nan,
        'group_review': np.nan,
        'seq': 342686,
        'crosswalk_parent_seq': 321982,
        'variance': np.nan,
        'effective_sample_size': 2000000.,
        'design_effect': np.nan,
        'is_outlier': 0,
        'standardized.case.definition': '',
        'serum_plasma': np.nan,
        'orig_source': '',
        'age_split': 0,
        'standard_error_orig': 3e-06,
        'mean_orig': 4e-05,
        'input_type': '',
        'uncertainty_type': '',
        'underlying_field_citation_value': np.nan
    }, index=[0])
    return cv


@pytest.fixture(scope='session')
def csmr(Demographics, ihme):
    csmr = CSMR(process_version_id=None, cause_id=587, demographics=Demographics,
                decomp_step='step3', gbd_round_id=6)
    csmr.raw = pd.DataFrame({
        'age_group_id': 2,
        'cause_id': 587,
        'location_id': 70,
        'measure_id': 1,
        'metric_id': 3,
        'sex_id': 2,
        'year_id': 1990,
        'acause': '',
        'age_group_name': '',
        'cause_name': '',
        'expected': False,
        'location_name': 'Canada',
        'measure_name': '',
        'metric_name': '',
        'sex': 'Female',
        'val': 5e-06,
        'upper': 6e-06,
        'lower': 2e-06
    }, index=[0])
    return csmr


@pytest.fixture(scope='session')
def asdr(Demographics, ihme):
    asdr = ASDR(demographics=Demographics, decomp_step='step3',
                gbd_round_id=6)
    asdr.raw = pd.DataFrame({
        'age_group_id': 2.0,
        'location_id': 70.0,
        'year_id': 1990.0,
        'sex_id': 2.0,
        'run_id': 84.0,
        'mean': 0.17,
        'upper': 0.19,
        'lower': 0.15
    }, index=[0])
    return asdr


@pytest.fixture(scope='session')
def population(Demographics, ihme):
    pop = Population(demographics=Demographics, decomp_step='step3',
                     gbd_round_id=6)
    pop.raw = pd.DataFrame({
        'age_group_id': 2.0,
        'location_id': 70.0,
        'year_id': 1990.0,
        'sex_id': 2.0,
        'population': 3711.,
        'run_id': np.nan
    }, index=[0])
    return pop


@pytest.fixture(scope='session')
def covariate_data(Demographics, ihme):
    cov = CovariateData(covariate_id=28, demographics=Demographics, decomp_step='step3', gbd_round_id=6)
    cov.raw = pd.DataFrame({
        'model_version_id': 28964,
        'covariate_id': 28,
        'covariate_name_short': 'ANC4_coverage_prop',
        'location_id': 70,
        'location_name': 'Canada',
        'year_id': 1990,
        'age_group_id': 22,
        'age_group_name': 'All Ages',
        'sex_id': 3,
        'sex': 'Both',
        'mean_value': 0.96,
        'lower_value': 0.96,
        'upper_value': 0.96
    }, index=[0])
    return cov


@pytest.fixture(scope='session')
def context(temp_directory):
    c = Context(model_version_id=0, make=True, configure_application=False,
                root_directory=temp_directory)
    return c


@pytest.fixture(scope='session')
def settings():
    return load_settings(BASE_CASE)


@pytest.fixture(scope='session')
def mi(asdr, cv, csmr, population, covariate_data, Demographics, settings):
    m = MeasurementInputsFromSettings(settings=settings)
    m.asdr = deepcopy(asdr)
    m.csmr = deepcopy(csmr)
    m.data = deepcopy(cv)
    m.covariate_data = [deepcopy(covariate_data)]
    m.population = deepcopy(population)
    m.demographics = Demographics
    m.configure_inputs_for_dismod(settings=settings, mortality_year_reduction=1)
    return m


@pytest.fixture(scope='session')
def dismod_data(mi, settings):
    return mi.dismod_data
