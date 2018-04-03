import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xp.config as cfg
from xp.simulate import Simulation


def run_and_check(x, scenario):
    instance = scenario(x)
    simu = Simulation(instance)
    simu.run()
    return instance.succeeded()


class ScenarioRobustnessEstimator:
    def __init__(self, scenario):
        self.scenario = scenario
        self.estimator = None

    def eval(self, samples):
        return self.estimator.decision_function(samples)

    def train(self, n_samples, verbose=False):
        scenario = self.scenario
        if verbose:
            print("Sampling the design space")
        samples = scenario.sample_valid(n_samples, max_trials=3*n_samples,
                                        rule='R')

        if verbose:
            print("Evaluating each scenario instance")
        res = Parallel(n_jobs=cfg.NCORES)(
            delayed(run_and_check)(sample, scenario) for sample in samples
        )

        if verbose:
            print("Training the classifier")
        samples_train, samples_test, res_train, res_test = train_test_split(
            samples, res, test_size=.5, random_state=n_samples, stratify=res
        )
        pipeline = make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', random_state=n_samples, cache_size=512),
                )
        C_range = np.logspace(*cfg.SVC_C_RANGE)
        gamma_range = np.logspace(*cfg.SVC_GAMMA_RANGE)
        class_weight_options = [None, 'balanced']
        param_grid = {
                'svc__gamma': gamma_range,
                'svc__C': C_range,
                'svc__class_weight': class_weight_options
                }
        grid = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=cfg.NCORES)
        grid.fit(samples_train, res_train)
        if verbose:
            print("The best parameters are {}".format(grid.best_params_))
            print("Score on the training set: {}".format(grid.best_score_))
            test_score = grid.score(samples_test, res_test)
            print("Score on the test set: {}".format(test_score))
        self.estimator = grid.best_estimator_
