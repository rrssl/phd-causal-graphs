import numpy as np
import scipy.optimize as opt


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


def maximize_robustness(scenario, estimators, x0, smin_coeff=1):
    energy = RobustnessEnergy(estimators, smin_coeff)
    constraint = dict(type='ineq', fun=PhysicalValidityConstraint(scenario))
    ndims = len(scenario.design_space)
    bounds = [[0, 1]] * ndims
    res = opt.minimize(energy, x0, method='SLSQP',
                       bounds=bounds, constraints=constraint,
                       options=dict(disp=True))
    return res
