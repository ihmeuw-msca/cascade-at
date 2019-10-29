from cascade_at.inputs.utilities.covariate_specifications import create_covariate_specifications
from cascade_at.inputs.utilities.gbd_ids import get_study_level_covariate_ids, get_country_level_covariate_ids
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class CovariateSpecs:
    def __init__(self, country_covariates):
        """

        :param country_covariates: (cascade_at.settings.settings.Configuration.country_covariate)
        """
        self.country_covariates = country_covariates
        self.covariate_multipliers, self.covariate_specs = create_covariate_specifications(
            country_covariate=self.country_covariates
        )
        self.country_covariate_ids = {
            spec.covariate_id for spec in self.covariate_specs
            if spec.study_country == "country"
        }
        self.study_id_to_name = get_study_level_covariate_ids()
        self.country_id_to_name = get_country_level_covariate_ids(list(self.country_covariate_ids))

        for cov in self.covariate_specs:
            if cov.study_country == 'study':
                short = self.study_id_to_name.get(cov.covariate_id, None)
            elif cov.study_country == 'country':
                short = self.country_id_to_name.get(cov.covariate_id, None)
            else:
                raise RuntimeError("Must be either study or country covariates.")
            if short is None:
                raise RuntimeError(f"Covariate {cov} is not found in id-to-name mapping.")
            cov.untransformed_covariate_name = short
