import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

import core.config as cfg


class MultivariateUniform:
    def __init__(self, ndims, a=0., b=1.):
        self.ndims = ndims
        self.a = a
        self.b = b

    def sample(self, n):
        return (self.b - self.a) * np.random.sample((n, self.ndims)) + self.a


class MultivariateMixtureOfGaussians:
    def __init__(self, mixture_params, weights=None):
        self.mixture_params = mixture_params
        self.weights = weights

    def sample(self, n):
        mps = self.mixture_params
        choice = np.random.choice(len(mps), size=n, replace=True,
                                  p=self.weights)
        normal = np.random.multivariate_normal
        return np.array([normal(mps[i][0], mps[i][1]) for i in choice])


def find_physically_valid_samples(scenario, distribution, n_valid, max_trials):
    """Find physically valid samples for this scenario.

    Parameters
    ----------
    scenario : scenario.Scenario
      Abstract scenario.
    distribution : {MultivariateUniform, MultivariateMixtureOfGaussians}
      Instance of the distribution to use for sampling.
    n_valid : int
      Number of expected valid samples.
    max_trials : int
      Maximum number of samples to try.

    Returns
    -------
    samples : sequence
      Physically valid samples. Size = (n,ndims), with n <= n_valid.

    """
    cand_samples = distribution.sample(max_trials)
    samples = []
    for sample in cand_samples:
        if scenario.check_physically_valid_sample(sample):
            samples.append(sample)
            if len(samples) == n_valid:
                break
    else:
        print("Attempt to find valid samples ran out of trials")
    return samples


def _simulate_and_get_nse(scenario, sample, **simu_kw):
    instance = scenario.instantiate_from_sample(sample, geom=None, phys=True,
                                                verbose_causal_graph=False)
    instance.simulate(**simu_kw)
    return len(instance.embedded_causal_graph.get_successful_events())


def _simulate_and_get_success(scenario, sample, **simu_kw):
    instance = scenario.instantiate_from_sample(sample, geom=None, phys=True,
                                                verbose_causal_graph=False)
    return instance.simulate(**simu_kw)


def find_successful_samples_uniform(scenario, n_succ=20, n_0=100, n_k=10,
                                    k_max=100, totals=None, **simu_kw):
    ndims = len(scenario.design_space)
    # Initialization
    samples = find_physically_valid_samples(
        scenario, MultivariateUniform(ndims), n_0, 100*n_0
    )
    labels = [_simulate_and_get_success(scenario, s, **simu_kw)
              for s in samples]
    # Main loop
    k = 0
    while k < k_max:
        total = sum(labels)
        print("Number of successful samples at step {}: {}".format(k, total))
        if totals is not None:
            totals.append(total)
        if total >= n_succ:
            break
        k += 1
        samples_k = find_physically_valid_samples(
            scenario, MultivariateUniform(ndims), n_k, 100*n_k
        )
        samples += samples_k
        labels += [_simulate_and_get_success(scenario, s, **simu_kw)
                   for s in samples_k]
    return samples, labels


def find_successful_samples_adaptive(scenario, n_succ=20, n_0=100, n_k=10,
                                     k_max=100, sigma=.01, totals=None,
                                     **simu_kw):
    """Sample the design space until enough successful samples are found.

    Returns
    -------
    samples : (n,n_dims) sequence
      All physically valid samples accumulated during the process.
    label : (n,) sequence
      Success label for each sample (True = success, False = failure).

    """
    ndims = len(scenario.design_space)
    nevents = len(scenario.causal_graph)
    cov = sigma * np.eye(ndims)
    # Initialization
    samples = find_physically_valid_samples(
        scenario, MultivariateUniform(ndims), n_0, 100*n_0
    )
    nse = [_simulate_and_get_nse(scenario, s, **simu_kw) for s in samples]
    labels = [nse_i == nevents for nse_i in nse]
    # Main loop
    k = 0
    while k < k_max:
        total = sum(labels)
        print("Number of successful samples at step {}: {}".format(k, total))
        if totals is not None:
            totals.append(total)
        if total >= n_succ:
            break
        k += 1
        # Select the top n_succ samples (or n_samples, whichever is smaller).
        n_top = min(n_succ, len(samples))
        top_ind = np.argpartition(-np.array(nse), n_top-1)[:n_top]
        top_samples = [samples[i] for i in top_ind]
        top_nse = [nse[i] for i in top_ind]
        # Compute their PMF.
        weights = np.array(top_nse, dtype=np.float64)
        weights /= weights.sum()
        # Generate the new samples.
        mixture_params = [(ts, cov) for ts in top_samples]
        dist = MultivariateMixtureOfGaussians(mixture_params, weights)
        samples_k = find_physically_valid_samples(scenario, dist, n_k, 100*n_k)
        samples += samples_k
        nse_k = [_simulate_and_get_nse(scenario, s, **simu_kw)
                 for s in samples_k]
        nse += nse_k
        labels += [nse_ki == nevents for nse_ki in nse_k]
    return samples, labels


def train_svc(samples, values, probability=False, dims=None, ret_score=False,
              verbose=True):
    samples = np.asarray(samples)
    if verbose:
        print("Number of samples:", samples.shape[0])
        print("Number of features:", samples.shape[1])
    # Create pipeline.
    steps = [
        StandardScaler(),
        SVC(kernel='rbf', probability=probability,
            random_state=len(samples), cache_size=512),
    ]
    if dims is not None:
        selector = FunctionTransformer(np.take,
                                       kw_args=dict(indices=dims, axis=1))
        steps.insert(0, selector)
    pipeline = make_pipeline(*steps)
    # Initialize cross-validation.
    C_range = np.logspace(*cfg.SVC_C_RANGE)
    gamma_range = np.logspace(*cfg.SVC_GAMMA_RANGE)
    class_weight_options = [None, 'balanced']
    param_grid = {
        'svc__gamma': gamma_range,
        'svc__C': C_range,
        'svc__class_weight': class_weight_options
    }
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5,
                        n_jobs=cfg.NCORES, iid=False)
    # Run cross-validation.
    grid.fit(samples, values)
    if verbose:
        print("The best parameters are {}".format(grid.best_params_))
        print("Score on the training set: {}".format(grid.best_score_))
    if ret_score:
        return grid.best_estimator_, grid.best_score_
    else:
        return grid.best_estimator_


def train_and_resample(scenario, init_samples, init_labels, resampler,
                       accuracy=.9, n_k=10, k_max=10, dims=None,
                       step_data=None, **simu_kw):
    """
    Train the SVC and add more samples until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    # Initialization
    samples = list(init_samples)
    labels = list(init_labels)
    # Main loop
    print("Running the train-and-resample loop")
    k = 0
    while k < k_max:
        k += 1
        # Train the SVC.
        X = np.asarray(samples)
        y = np.asarray(labels)
        estimator, score = train_svc(X, y, dims=dims, ret_score=True)
        if step_data is not None:
            step_data.append((X, y, estimator, score))
        if score >= accuracy:
            break
        # Generate the new samples.
        samples_k, labels_k = resampler(scenario, X, y, estimator, n_k, dims,
                                        **simu_kw)
        if samples_k and labels_k:
            samples.extend(samples_k)
            labels.extend(labels_k)
        else:
            break
    # Calibrate
    print("Calibrating the classifier")
    estimator = train_svc(samples, labels, probability=True, dims=dims,
                          verbose=False)
    return estimator


def _sample_uniform_and_run(scenario, X, y, estimator, n, dims=None,
                            **simu_kw):
    if dims is not None:
        random_success = X[np.random.choice(np.flatnonzero(y))]
        a = random_success.copy()
        b = a.copy()
        a[dims] = 0
        b[dims] = 1
        dist = MultivariateUniform(X.shape[1], a, b)
    else:
        dist = MultivariateUniform(X.shape[1])
    samples = find_physically_valid_samples(scenario, dist, n, 100*n)
    labels = [_simulate_and_get_success(scenario, s, **simu_kw)
              for s in samples]
    return samples, labels


def train_and_add_uniform_samples(scenario, init_samples, init_labels,
                                  accuracy=.9, n_k=10, k_max=10, dims=None,
                                  step_data=None, **simu_kw):
    """
    Train the SVC and add more uniform samples until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    return train_and_resample(
        scenario, init_samples, init_labels, _sample_uniform_and_run,
        accuracy, n_k, k_max, dims,
        step_data, **simu_kw
    )


def _sample_misclassified_and_run(scenario, X, y, estimator, n, dims=None,
                                  **simu_kw):
    is_wrong = estimator.predict(X) != y
    if not is_wrong.any():
        return [], []
    wrong_X = X[is_wrong]
    wrong_af = np.abs(estimator.decision_function(wrong_X))
    # Compute weights.
    weights = wrong_af / wrong_af.sum()
    # Generate samples.
    diag = 1 / estimator.named_steps['standardscaler'].scale_
    if dims is not None:
        # Restore the full-sized diagonal (where non-free dims are 0)
        fulldiag = np.zeros(X.shape[1])
        fulldiag[dims] = diag
        diag = fulldiag
    scale = np.diagflat(diag)
    mixture_params = [(xi, afi*scale)
                      for xi, afi in zip(wrong_X, wrong_af)]
    dist = MultivariateMixtureOfGaussians(mixture_params, weights)
    samples = find_physically_valid_samples(scenario, dist, n, 100*n)
    labels = [_simulate_and_get_success(scenario, s, **simu_kw)
              for s in samples]
    return samples, labels


def train_and_consolidate_boundary(scenario, init_samples, init_labels,
                                   accuracy=.9, n_k=10, k_max=10, dims=None,
                                   step_data=None, **simu_kw):
    """
    Train the SVC and consolidate the boundary around its misclassified samples
    until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    return train_and_resample(
        scenario, init_samples, init_labels, _sample_misclassified_and_run,
        accuracy, n_k, k_max, dims,
        step_data, **simu_kw
    )


def _sample_support_and_run(scenario, X, y, estimator, n, dims=None,
                            **simu_kw):
    # Retrieve the support vectors.
    support = estimator.named_steps['svc'].support_
    # Compute weights.
    af = np.abs(estimator.decision_function(X[support]))
    weights = af / af.sum()
    # Generate samples.
    diag = 1 / estimator.named_steps['standardscaler'].scale_
    if dims is not None:
        # Restore the full-sized diagonal (where non-free dims are 0)
        fulldiag = np.zeros(X.shape[1])
        fulldiag[dims] = diag
        diag = fulldiag
    scale = np.diagflat(diag)
    mixture_params = [(X[si], afi*scale)
                      for si, afi in zip(support, af)]
    dist = MultivariateMixtureOfGaussians(mixture_params, weights)
    samples = find_physically_valid_samples(scenario, dist, n, 100*n)
    labels = [_simulate_and_get_success(scenario, s, **simu_kw)
              for s in samples]
    return samples, labels


def train_and_consolidate_boundary2(scenario, init_samples, init_labels,
                                    accuracy=.9, n_k=10, k_max=10, dims=None,
                                    step_data=None, **simu_kw):
    """
    Train the SVC and consolidate the boundary around its support vectors until
    accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    return train_and_resample(
        scenario, init_samples, init_labels, _sample_support_and_run,
        accuracy, n_k, k_max, dims,
        step_data, **simu_kw
    )
