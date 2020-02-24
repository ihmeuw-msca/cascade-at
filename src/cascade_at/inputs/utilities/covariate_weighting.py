import numpy as np
from intervaltree import IntervalTree

from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def values(interval):
    return interval.begin, interval.end, interval.data


def interval_weighting(intervals, lower, upper):
    """
    Compute a weighting function by finding the proportion
    within the dataframe df's lower and upper bounds.

    Note: intervals is of the form ((lower, upper, id), ...)
    """

    if len(intervals) == 1:
        return np.asarray([1])
    wts = np.ones(len(intervals))
    lower_limit, upper_limit = intervals[0], intervals[-1]
    wts[0] = (lower_limit[1] - lower) / np.diff(lower_limit[:2])
    wts[-1] = (upper - upper_limit[0]) / np.diff(upper_limit[:2])
    return wts


class CovariateInterpolator:
    def __init__(self,
                 covariate,
                 population):
        """
        Interpolates a covariate by population weighting.
        :param covariate: (pd.DataFrame)
        :param population: (pd.DataFrame)
        """
        # Covariates must be sorted by both age_group_id and age_lower because age_lower is not unique to age_group_id
        indices = ['location_id', 'sex_id', 'year_id', 'age_group_id']
        sort_order = indices + ['age_lower']

        self.covariate = covariate.sort_values(by=sort_order)
        self.population = population.sort_values(by=sort_order)

        self.location_ids = self.covariate.location_id.unique()

        self.age_intervals = IntervalTree.from_tuples(
            self.covariate[['age_lower', 'age_upper', 'age_group_id']].values
        )
        self.time_intervals = IntervalTree.from_tuples([
            (t, t+1, t) for t in self.covariate.year_id.unique()
        ])

        self.dict_cov = dict(zip(
            map(tuple, self.covariate[indices].values.tolist()), self.covariate['mean_value'].values
        ))
        self.dict_pop = dict(zip(
            map(tuple, self.population[indices].values.tolist()), self.population['population'].values
        ))

    def _weighting(self, age_lower, age_upper, time_lower, time_upper):
        age_groups = sorted(map(values, self.age_intervals[age_lower: age_upper]))
        age_group_ids = [a[-1] for a in age_groups]
        age_wts = interval_weighting(tuple(age_groups), age_lower, age_upper)

        time_groups = sorted(map(values, self.time_intervals[time_lower: time_upper]))
        year_ids = [t[-1] for t in time_groups]
        time_wts = interval_weighting(tuple(time_groups), time_lower, time_upper)

        # The order of outer must agree with the covariate and population sort order
        wt = np.outer(time_wts, age_wts)
        return age_group_ids, year_ids, wt

    def interpolate(self, loc_id, sex_id, age_lower, age_upper, time_lower, time_upper):
        """
        Main interpolation function.
        """
        if loc_id not in self.location_ids:
            LOG.warn(f"Covariate is missing for location_id {loc_id},"
                     f"sex_id {sex_id} -- setting the value to None.")
            cov_value = None
        else:
            age_group_ids, year_ids, epoch_weights = self._weighting(
                age_lower=age_lower, age_upper=age_upper,
                time_lower=time_lower, time_upper=time_upper
            )
            shape = epoch_weights.shape
            # This loop indexing order matters, and must agree with the covariate and population sort order
            cov_value = np.asarray([self.dict_cov[(loc_id, sex_id, year_id, age_id)]
                                    for year_id in year_ids for age_id in age_group_ids]).reshape(shape)
            # This loop indexing order matters, and must agree with the covariate and population sort order
            pop_value = np.asarray([self.dict_pop[(loc_id, sex_id, year_id, age_id)]
                                    for year_id in year_ids for age_id in age_group_ids]).reshape(shape)

            weight = epoch_weights * pop_value
            cov_value = np.average(cov_value, weights=weight)
        return cov_value


def get_interpolated_covariate_values(data_df, covariate_dict,
                                      population_df):
    """
    Gets the unique age-time combinations from the data_df, and creates
    interpolated covariate values for each of these combinations by population-weighting
    the standard GBD age-years that span the non-standard combinations.

    :param data_df: (pd.DataFrame)
    :param covariate_dict: Dict[pd.DataFrame] with covariate names as keys
    :param population_df: (pd.DataFrame)
    :return: pd.DataFrame
    """
    data = data_df.copy()
    pop = population_df.copy()

    data_groups = data.groupby([
        'location_id', 'sex_id', 'age_lower', 'age_upper', 'time_lower', 'time_upper'
    ], as_index=False)

    cov_objects = {cov_name: CovariateInterpolator(covariate=raw_cov, population=pop)
                   for cov_name, raw_cov in covariate_dict.items()}
    num_groups = len(data_groups)
    for i, (k, v) in enumerate(data_groups):
        if i % 1000 == 0:
            LOG.info(f"Processed {i} of {num_groups} data groups.")
        [loc_id, sex_id, age_lower, age_upper, time_lower, time_upper] = k
        for cov_id, cov_obj in cov_objects.items():
            cov_value = cov_obj.interpolate(
                loc_id=loc_id, sex_id=sex_id,
                age_lower=age_lower, age_upper=age_upper,
                time_lower=time_lower, time_upper=time_upper
            )
            data.loc[v.index, cov_id] = cov_value
    return data
