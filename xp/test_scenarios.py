import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(".."))
import xp.scenarios as scenarios  # noqa: E402


def main():
    for sc in scenarios.SCENARIOS:
        print("Testing {}".format(sc))
        params = sc.sample_valid(1)[0]
        kwargs = {}
        if sc in (scenarios.DominoesStraightLastFree,
                  scenarios.DominoesStraightTwoLastFree):
            kwargs['nprev'] = 2
        if sc is scenarios.DominoesStraightLastFree and kwargs['nprev'] == 0:
            params[-1] = 0
        if sc is scenarios.BallPlankDominoes:
            kwargs['ndoms'] = 2
        scene = sc.init_scenario(params, **kwargs)[0]
        back_params = sc.get_parameters(scene)
        assert np.allclose(params, back_params), params - back_params
    print("All tests OK")


if __name__ == "__main__":
    main()
