import pytest

from cascade_at.dismod.api.run_dismod import run_dismod
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


def test_dmdismod_init(temp_directory, dismod):
    run = run_dismod(dm_file=str(temp_directory / 'temp.db'), command=['init'])
    if run.exit_status:
        print(run.stdout)
        print(run.stderr)
    assert run.exit_status == 0
