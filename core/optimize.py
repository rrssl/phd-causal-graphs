import numpy as np
import scipy.optimize as opt

from core.config import NCORES
from core.robustness import compute_label


class RobustnessEnergy:
    def __init__(self, estimators, smin_coeff=1):
        self.estimators = estimators
        self.smin_coeff = smin_coeff

    def __call__(self, x):
        x = x.reshape(1, -1)
        r = np.array([e.predict_proba(x)[0, 1] for e in self.estimators])
        exp = np.exp(-self.smin_coeff * r)
        return -r.dot(exp) / exp.sum()


class PhysicalValidityConstraint:
    def __init__(self, scenario):
        self.scenario = scenario

    def __call__(self, x):
        return self.scenario.instantiate_from_sample(
            x, geom=None, phys=True, verbose_causal_graph=False
        ).scene.get_physical_validity_constraint()


class SuccessConstraint:
    def __init__(self, scenario, **simu_kw):
        self.scenario = scenario
        self.simu_kw = simu_kw

    def __call__(self, x):
        if self.scenario.check_physically_valid_sample(x):
            # _, labels = compute_label(
            #     self.scenario, x, ret_events_labels=True, **self.simu_kw
            # )
            # return sum(filter(None, labels.values())) / len(labels) - 1.
            return compute_label(self.scenario, x, **self.simu_kw) - 1.
        else:
            return 0.


def maximize_robustness(scenario, estimators, x0, smin_coeff=1, **simu_kw):
    energy = RobustnessEnergy(estimators, smin_coeff)
    phys_cs = dict(type='ineq', fun=PhysicalValidityConstraint(scenario))
    succ_cs = dict(type='ineq', fun=SuccessConstraint(scenario, **simu_kw))
    ndims = len(scenario.design_space)
    bounds = [(0, 1)] * ndims
    res = opt.minimize(energy, x0, method='SLSQP',
                       bounds=bounds, constraints=(phys_cs, succ_cs),
                       options=dict(disp=True))
    return res
