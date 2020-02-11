"""
Helper functions for Model and ConstructModel
"""

from copy import deepcopy
import numpy as np

from cascade_at.model.covariate import Covariate
from cascade_at.model.var import Var
from cascade_at.model.smooth_grid import SmoothGrid
from cascade_at.model.priors import Constant
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def rectangular_data_to_var(gridded_data):
    """Using this very regular data, where every age and time is present,
    construct an initial guess as a Var object. Very regular means that there
    is a complete set of ages-cross-times."""
    gridded_data = gridded_data.copy()
    try:
        gridded_data['age'] = gridded_data[['age_lower', 'age_upper']].mean(axis=1)
        gridded_data['time'] = gridded_data[['time_lower', 'time_upper']].mean(axis=1)
    except AttributeError:
        LOG.error(f"Data to make a var has columns {gridded_data.columns}")
        raise RuntimeError(
            f"Wrong columns in rectangular_data_to_var {gridded_data.columns}")
    gridded_data = gridded_data.sort_values(by=['age', 'time'])
    guess = Var(ages=sorted(gridded_data['age'].unique()), times=sorted(gridded_data['time'].unique()))
    assert guess.variable_count() == len(gridded_data), \
        "Number of age/time points exceed number of unique age/time points"
    guess[:, :] = gridded_data['mean'].values.reshape((-1, 1))
    return guess


def constraint_from_rectangular_data(rate_var, default_age_time):
    """Takes data on a complete set of ages and times, makes a constraint grid.

    Args:
        rate_var: A function of age and time to represent a rate.
        default_age_time:
    """
    omega_grid = SmoothGrid(ages=default_age_time["age"], times=default_age_time["time"])
    for age, time in omega_grid.age_time():
        omega_grid.value[age, time] = Constant(rate_var(age, time))
    return omega_grid


def smooth_grid_from_smoothing_form(default_age_time, single_age_time, smooth):
    """
    Create a new SmoothGrid from the settings in EpiViz-AT at the Smoothing
    level.

    Args:
        default_age_time (List[ages, times]): Two members, the ages and the time.
        single_age_time (List[float]): Two members, an age and a time.
        smooth (cascade.input_data.configuration.form.Smoothing): The form element.
    Returns:
        SmoothGrid: A new smooth grid.
    """
    ages, times = construct_grid_ages_times(default_age_time, single_age_time, smooth)
    rate_grid = SmoothGrid(ages=ages, times=times)
    for kind in ["value", "dage", "dtime"]:
        if not smooth.is_field_unset("default") and not smooth.default.is_field_unset(kind):
            getattr(rate_grid, kind)[:, :] = getattr(smooth.default, kind).prior_object
        if not smooth.is_field_unset("mulstd") and not smooth.mulstd.is_field_unset(kind):
            getattr(rate_grid, kind).mulstd_prior = getattr(smooth.mulstd, kind).prior_object
    if not smooth.is_field_unset("detail"):
        for smoothing_prior in smooth.detail:
            for a, t in matching_knots(rate_grid, smoothing_prior):
                getattr(rate_grid, smoothing_prior.prior_type)[a, t] = smoothing_prior.prior_object
    return rate_grid


def matching_knots(rate_grid, smoothing_prior):
    """
    Get lower and upper out of the smoothing prior. This uses the age, time,
    and "born" lower and upper bounds to return
    ages and times in the grid that are within those bounds.
    The goal is to apply a prior selectively to those knots.

    Args:
        rate_grid:
        smoothing_prior (cascade.input_data.configuration.form.SmoothingPrior):
            A single smoothing prior.

    Returns:
        Iterator over (a, t) that match. Can be nothing.
    """
    extents = dict()
    for extent in ["age", "time", "born"]:
        extents[extent] = np.zeros(2, dtype=np.float)
        for side_idx, side, default_extent in [(0, "lower", -np.inf), (1, "upper", np.inf)]:
            name = f"{extent}_{side}"
            if smoothing_prior.is_field_unset(name):
                extents[extent][side_idx] = default_extent
            else:
                extents[extent][side_idx] = getattr(smoothing_prior, name)
    # `np.meshgrid` generates every combination of age and time as two numpy arrays.
    ages, times = np.meshgrid(rate_grid.ages, rate_grid.times)
    assert ages.shape == (len(rate_grid.times), len(rate_grid.ages))
    assert times.shape == (len(rate_grid.times), len(rate_grid.ages))
    in_age = (ages >= extents["age"][0]) & (ages <= extents["age"][1])
    in_time = (times >= extents["time"][0]) & (times <= extents["time"][1])
    in_born = (ages <= times - extents["born"][0]) & (ages >= times - extents["born"][1])
    cover = in_age & in_time & in_born
    if not np.any(cover):
        LOG.info(f"No ages and times match prior with extents {extents}.")
    yield from zip(ages[cover], times[cover])


def construct_grid_ages_times(default_age_time, single_age_time, smooth):
    if not smooth.is_field_unset("age_time_specific") and smooth.age_time_specific == 0:
        return single_age_time

    ages = smooth.age_grid
    if ages is None:
        # hasattr because this may be a Smoothing form or a Covariate form.
        if hasattr(smooth, "rate") and smooth.rate == "pini":
            ages = np.array([0], dtype=np.float)
        else:
            ages = default_age_time["age"]
    else:
        ages = np.sort(np.array(ages, dtype=np.float))
    times = smooth.time_grid
    if times is None:
        times = default_age_time["time"]
    else:
        times = np.sort(np.array(times, dtype=np.float))
    return ages, times


def construct_model_covariates(default_age_time, single_age_time, covariate_multipliers, model):
    """The covariate multipliers are of all types: alpha, beta, and gamma. This adds
    their priors to the Model.

    Args:
        default_age_time (Tuple[ndarray, ndarray]): ages and times
        single_age_time (float, float): The single age and time to use if it's
            a point value.
        covariate_multipliers (List[EpiVizCovariateMultiplier): A list of specifications
            for covariate multipliers. This assumes data has already been read,
            because that data determines names for the multipliers.
    """
    for mulcov in covariate_multipliers:
        grid = smooth_grid_from_smoothing_form(default_age_time, single_age_time, mulcov.grid_spec)
        model[mulcov.group][mulcov.key] = grid


def covariates_list(covariate_data_spec):
    covariate_list = list()
    for c in covariate_data_spec:
        LOG.debug(f"Adding covariate reference {c.name}.reference={c.reference}")
        covariate_list.append(Covariate(c.name, c.reference, c.max_difference))
    return covariate_list


def integrand_grids(alchemy, integrands):
    """
    Get the age-time grids associated with a list of integrands.
    Should be used for converting priors to posteriors.

    Args:
        alchemy:
        integrands:

    Returns:

    """
    grids = dict()

    default_grid = alchemy.construct_age_time_grid()
    for integrand in integrands:
        grids[integrand] = deepcopy(default_grid)

    rate_grids = alchemy.get_all_rates_grids()
    for k, v in rate_grids.items():
        if k in integrands:
            grids[k].update({'age': v.ages})
            grids[k].update({'time': v.times})
    return grids
