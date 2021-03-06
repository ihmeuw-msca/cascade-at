from collections import defaultdict
import itertools as it

import numpy as np

from cascade_at.core.log import get_loggers
from cascade_at.model.model import Model
from cascade_at.model.utilities.grid_helpers import smooth_grid_from_smoothing_form
from cascade_at.model.utilities.grid_helpers import rectangular_data_to_var
from cascade_at.model.utilities.grid_helpers import constraint_from_rectangular_data

LOG = get_loggers(__name__)


class Alchemy:
    def __init__(self, settings):
        """
        An object initialized with model settings from
        cascade.settings.configuration.Configuration that can be used
        to construct parent-child location-specific models with
        the attribute ModelConstruct.construct_two_level_model().

        Parameters:
            settings (cascade_at.settings.settings.Configuration):
        
        Usage:
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings
        >>> from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
        
        >>> settings = load_settings(BASE_CASE)
        >>> mc = Alchemy(settings)
        
        >>> i = MeasurementInputsFromSettings(settings)
        >>> i.get_raw_inputs()

        >>> mc.construct_two_level_model(location_dag=i.location_dag,
        >>>                              parent_location_id=102,
        >>>                              covariate_specs=i.covariate_specs)
        """
        self.settings = settings
        self.age_time_grid = self.construct_age_time_grid()
        self.single_age_time_grid = self.construct_single_age_time_grid()

        self.model = None

    def construct_age_time_grid(self):
        """
        Construct a DEFAULT age-time grid,
        to be updated when we initialize the model.

        :return:
        """
        default_age_time = dict()
        default_age_time["age"] = np.linspace(0, 100, 21)
        default_age_time["time"] = np.linspace(1990, 2015, 6)

        for kind in ["age", "time"]:
            default_grid = getattr(self.settings.model, f"default_{kind}_grid")
            if default_grid is not None:
                default_age_time[kind] = np.sort(np.array(default_grid, dtype=np.float))

        return default_age_time

    def construct_single_age_time_grid(self):
        """
        Construct a single age-time grid.
        Use this age and time when a smooth grid doesn't depend on age and time.

        :return:
        """
        single_age = self.age_time_grid["age"][:1]
        single_time = [self.age_time_grid["time"][len(self.age_time_grid["time"]) // 2]]
        single_age_time = (single_age, single_time)
        return single_age_time

    def get_smoothing_grid(self, rate):
        """
        Construct a smoothing grid for any rate in the model.

        Parameters:
            rate: (cascade_at.settings.settings_configuration.Smoothing)

        Returns: (cascade_at.model.smooth_grid.SmoothGrid)

        """
        return smooth_grid_from_smoothing_form(
            default_age_time=self.age_time_grid,
            single_age_time=self.single_age_time_grid,
            smooth=rate
        )

    def get_all_rates_grids(self):
        """
        Get a dictionary of all the rates and their grids in the model.

        Returns: dict[str: SmoothGrid]

        """
        return {c.rate: self.get_smoothing_grid(rate=c) for c in self.settings.rate}

    @staticmethod
    def estimate_grid_parameters(grid_priors, draws, ages, times):
        """
        Estimates using MLE the parameters for the grid using prior draws.
        Updates the grid_priors object in place, so returns nothing.

        Args:
            grid_priors: (cascade_at...)
            draws: (np.ndarray) 3-d array coming out of `DismodExtractor.gather_draws_for_prior_grid()`
            ages: (np.array)
            times: (np.array)
        """
        assert isinstance(draws, np.ndarray)
        assert len(draws.shape) == 3
        assert draws.shape[0] == len(ages), "Not the same number of ages in the prior as the grid"
        assert draws.shape[1] == len(times), "Not the same number of times in the prior as the grid"
        for age_idx, time_idx in it.product(range(len(ages)), range(len(times))):
            age = ages[age_idx]
            time = times[time_idx]
            grid_priors[age, time] = grid_priors[age, time].mle(draws[age_idx, time_idx, :])
    
    def construct_two_level_model(self, location_dag, parent_location_id, covariate_specs, weights=None,
                                  omega_df=None, update_prior=None):
        """
        Construct a Model object for a parent location and its children.

        Parameters:
            location_dag: (cascade.inputs.locations.LocationDAG)
            parent_location_id: (int)
            covariate_specs (cascade_at.inputs.covariate_specs.CovariateSpecs): covariate
                specifications, specifically will use covariate_specs.covariate_multipliers
            weights:
            omega_df: (pd.DataFrame)
            update_prior: (dict) of (dict)
        """
        children = list(location_dag.dag.successors(parent_location_id))
        model = Model(
            nonzero_rates=self.settings.rate,
            parent_location=parent_location_id,
            child_location=children,
            covariates=covariate_specs.covariate_list,
            weights=weights
        )

        # First construct the rate grid, and update with prior
        # information from a parent for value, dage, and dtime.
        for smooth in self.settings.rate:
            rate_grid = self.get_smoothing_grid(rate=smooth)
            if update_prior is not None:
                if smooth.rate in update_prior:
                    prior = update_prior[smooth.rate]
                    # Check that the prior grid lines up with this rate
                    # grid. If it doesn't, we have a problem.
                    assert (prior['ages'] == rate_grid.ages)
                    assert (prior['times'] == rate_grid.times)
                    # For each of the types of priors, update rate_grid
                    # with the new prior information from the update_prior
                    # object that has info from a different model fit
                    if 'value' in prior:
                        self.estimate_grid_parameters(
                            grid_priors=rate_grid.value, draws=prior['value'],
                            ages=rate_grid.ages, times=rate_grid.times
                        )
                    if 'dage' in prior:
                        self.estimate_grid_parameters(
                            grid_priors=rate_grid.dage, draws=prior['dage'],
                            ages=rate_grid.ages[:-1], times=rate_grid.times
                        )
                    if 'dtime' in prior:
                        self.estimate_grid_parameters(
                            grid_priors=rate_grid.dtime, draws=prior['dtime'],
                            ages=rate_grid.ages, times=rate_grid.times[:-1]
                        )
            model.rate[smooth.rate] = rate_grid
        
        # Second construct the covariate grids
        for mulcov in covariate_specs.covariate_multipliers:
            grid = smooth_grid_from_smoothing_form(
                    default_age_time=self.age_time_grid,
                    single_age_time=self.single_age_time_grid,
                    smooth=mulcov.grid_spec
                )
            model[mulcov.group][mulcov.key] = grid

        # Construct the random effect grids, based on the parent location
        # specified.
        if self.settings.random_effect:
            random_effect_by_rate = defaultdict(list)
            for smooth in self.settings.random_effect:
                re_grid = smooth_grid_from_smoothing_form(
                    default_age_time=self.age_time_grid,
                    single_age_time=self.single_age_time_grid,
                    smooth=smooth
                )
                if not smooth.is_field_unset("location") and smooth.location in model.child_location:
                    location = smooth.location
                else:
                    location = None
                model.random_effect[(smooth.rate, location)] = re_grid
                random_effect_by_rate[smooth.rate].append(location)

            for rate_to_check, locations in random_effect_by_rate.items():
                if locations != [None] and set(locations) != set(model.child_location):
                    raise RuntimeError(f"Random effect for {rate_to_check} does not have "
                                       f"entries for all child locations, only {locations} "
                                       f"instead of {model.child_location}.")

        # Lastly, constrain omega for the parent and the random effects for the children.
        if self.settings.model.constrain_omega:
            LOG.info("Adding the omega constraint.")
            
            if omega_df is None:
                raise RuntimeError("Need an omega data frame in order to constrain omega.")
            
            parent_omega = omega_df.loc[omega_df.location_id == parent_location_id].copy()
            if parent_omega.empty:
                raise RuntimeError("No omega values for location {parent_location_id}.")

            omega = rectangular_data_to_var(gridded_data=parent_omega)
            model.rate["omega"] = constraint_from_rectangular_data(
                rate_var=omega,
                default_age_time=self.age_time_grid
            )
            
            locations = set(omega_df.location_id.unique().tolist())
            children_without_omega = set(children) - set(locations)
            if children_without_omega:
                LOG.warning(f"Children of {parent_location_id} missing omega {children_without_omega}"
                            f"so not including child omega constraints")
            else:
                for child in children:
                    child_omega = omega_df.loc[omega_df.location_id == child].copy()
                    assert not child_omega.empty
                    child_rate = rectangular_data_to_var(gridded_data=child_omega)

                    def child_effect(age, time):
                        return np.log(child_rate(age, time) / omega(age, time))
                    
                    model.random_effect[("omega", child)] = constraint_from_rectangular_data(
                        rate_var=child_effect,
                        default_age_time=self.age_time_grid
                    )
        return model

