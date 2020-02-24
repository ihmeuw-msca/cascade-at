import pytest

import numpy as np
from cascade_at.dismod.api.fill_extract_helpers.utils import vec_to_midpoint


@pytest.mark.parametrize("array,mid", [
    ([1990, 1991, 1992], [1990.5, 1991.5]),
    ([0, 5, 10], [2.5, 7.5]),
    ([0, 0, 5, 10], [0, 2.5, 7.5])
])
def test_vec_to_midpoint(array, mid):
    np.testing.assert_array_equal(vec_to_midpoint(np.array(array)), np.array(mid))
