from itertools import compress

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xp.config as cfg
from xp.causal import EventState
from xp.simulate import Simulation


class RobustnessEstimator:
    def __init__(self, scenario, ids=None):
        self.scenario = scenario
        self.ids = ids if ids is not None else slice(None)
        self.estimator = None

    def eval(self, samples):
        samples = np.asarray(samples)[:, self.ids]
        return self.estimator.decision_function(samples)

    def _generate_samples(self, n_samples):
        samples = self.scenario.sample_valid(
            n_samples,
            max_trials=cfg.TRAINING_SAMPLING_FACTOR*n_samples,
            rule='R'
        )
        return samples

    def run_and_check(self, x):
        raise NotImplementedError

    def sample_and_train(self, n_samples, verbose=False):
        if verbose:
            print("Sampling the design space")
        samples = self._generate_samples(n_samples)
        if verbose:
            print("Evaluating each sample")
        res = Parallel(n_jobs=cfg.NCORES)(
            delayed(self.run_and_check)(sample) for sample in samples
        )
        valid = [r is not None for r in res]
        if not all(valid):
            samples = samples[valid]
            res = list(compress(res, valid))
        self.train_from_samples(samples, res, verbose)

    def train(self, samples, values, verbose=False):
        samples = np.asarray(samples)[:, self.ids]
        if verbose:
            print("Number of samples:", samples.shape[0])
            print("Number of features:", samples.shape[1])

        if verbose:
            print("Training the classifier")
        samples_train, samples_test, res_train, res_test = train_test_split(
            samples, values, test_size=.5, random_state=len(samples),
            stratify=values
        )
        pipeline = make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', random_state=len(samples), cache_size=512),
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


class FullScenarioRobustnessEstimator(RobustnessEstimator):
    def __init__(self, scenario):
        super().__init__(scenario)

    def run_and_check(self, x):
        instance = self.scenario(x)
        simu = Simulation(instance)
        simu.run()
        return instance.succeeded()

    # LEGACY
    def train(self, n_samples, verbose=False):
        super().sample_and_train(n_samples, verbose)


class EventRobustnessEstimator(RobustnessEstimator):
    answer = {
        EventState.success: True,
        EventState.failure: False,
        EventState.asleep: None
    }

    def __init__(self, scenario, event, ids=None):
        super().__init__(scenario, ids)
        self.event = event

    def run_and_check(self, x):
        instance = self.scenario(x)
        simu = Simulation(instance)
        simu.run()
        return self.answer[instance.causal_graph.get_event(self.event).state]
