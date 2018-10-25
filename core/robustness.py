import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import core.config as cfg


class MultivariateUniform:
    def __init__(self, ndims):
        self.ndims = ndims

    def sample(self, n):
        return np.random.sample((n, self.ndims))


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
        scenario, MultivariateUniform(ndims), n_0, 5*n_0
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
            scenario, MultivariateUniform(ndims), n_k, 5*n_k
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
        scenario, MultivariateUniform(ndims), n_0, 5*n_0
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
        samples_k = find_physically_valid_samples(scenario, dist, n_k, 5*n_k)
        samples += samples_k
        nse_k = [_simulate_and_get_nse(scenario, s, **simu_kw)
                 for s in samples_k]
        nse += nse_k
        labels += [nse_ki == nevents for nse_ki in nse_k]
    return samples, labels


def train_svc(samples, values, probability=False, verbose=True):
    samples = np.asarray(samples)
    if verbose:
        print("Number of samples:", samples.shape[0])
        print("Number of features:", samples.shape[1])

    if verbose:
        print("Training the classifier")
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=probability,
            random_state=len(samples), cache_size=512),
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
    grid.fit(samples, values)
    if verbose:
        print("The best parameters are {}".format(grid.best_params_))
        print("Score on the training set: {}".format(grid.best_score_))
    return grid.best_estimator_


def train_and_add_uniform_samples(scenario, init_samples, init_labels,
                                  accuracy=.9, n_k=10, k_max=10,
                                  step_data=None, **simu_kw):
    """
    Train the SVC and add more uniform samples until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    ndims = len(scenario.design_space)
    # Initialization
    samples = list(init_samples)
    labels = list(init_labels)
    # Main loop
    k = 0
    while k < k_max:
        k += 1
        # Train the SVC.
        X = np.asarray(samples)
        y = np.asarray(labels)
        estimator = train_svc(X, y)
        score = estimator.score(X, y)
        print("Total score:", score)
        if step_data is not None:
            step_data.append((X, y, estimator))
        if score >= accuracy:
            break
        # Generate the new samples.
        samples_k = find_physically_valid_samples(
            scenario, MultivariateUniform(ndims), n_k, 5*n_k
        )
        samples += samples_k
        labels += [_simulate_and_get_success(scenario, s, **simu_kw)
                   for s in samples_k]
    # Calibrate
    estimator = train_svc(samples, labels, probability=True)
    return estimator


def train_and_consolidate_boundary(scenario, init_samples, init_labels,
                                   accuracy=.9, n_k=10, k_max=10,
                                   step_data=None, **simu_kw):
    """
    Train the SVC and consolidate its boundary until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    # Initialization
    samples = list(init_samples)
    labels = list(init_labels)
    # Main loop
    k = 0
    while k < k_max:
        k += 1
        # Train the SVC.
        X = np.asarray(samples)
        y = np.asarray(labels)
        estimator = train_svc(X, y)
        scale = np.diagflat(1 / estimator.named_steps['standardscaler'].scale_)
        score = estimator.score(X, y)
        print("Total score:", score)
        if step_data is not None:
            step_data.append((X, y, estimator))
        if score >= accuracy:
            break
        # Retrieve the misclassified samples.
        f = estimator.decision_function(X)
        is_wrong = (f >= 0) != y
        wrong_X = X[is_wrong]
        wrong_f = f[is_wrong]
        # Compute their PMF.
        weights = abs(wrong_f)
        weights /= weights.sum()
        # Generate the new samples.
        mixture_params = [(ws, wsf*scale)
                          for ws, wsf in zip(wrong_X, abs(wrong_f))]
        dist = MultivariateMixtureOfGaussians(mixture_params, weights)
        samples_k = find_physically_valid_samples(scenario, dist, n_k, 5*n_k)
        samples += samples_k
        labels += [_simulate_and_get_success(scenario, s, **simu_kw)
                   for s in samples_k]
    # Calibrate
    estimator = train_svc(samples, labels, probability=True)
    return estimator


def train_and_consolidate_boundary2(scenario, init_samples, init_labels,
                                    accuracy=.9, n_k=10, k_max=10,
                                    step_data=None, **simu_kw):
    """
    Train the SVC and consolidate its boundary until accuracy is reached.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
      Trained estimator.

    """
    # ndims = len(scenario.design_space)
    # Initialization
    samples = list(init_samples)
    labels = list(init_labels)
    # Main loop
    k = 0
    while k < k_max:
        k += 1
        # Train the SVC.
        X = np.asarray(samples)
        y = np.asarray(labels)
        estimator = train_svc(X, y)
        scale = np.diagflat(1 / estimator.named_steps['standardscaler'].scale_)
        score = estimator.score(X, y)
        print("Total score:", score)
        if step_data is not None:
            step_data.append((X, y, estimator))
        if score >= accuracy:
            break
        # Retrieve the support vectors.
        support = estimator.named_steps['svc'].support_
        f = estimator.decision_function(X[support])
        # Compute their PMF.
        weights = abs(f)
        weights /= weights.sum()
        # Generate the new samples.
        mixture_params = [(s, sf*scale)
                          for s, sf in zip(X[support], abs(f))]
        dist = MultivariateMixtureOfGaussians(mixture_params, weights)
        samples_k = find_physically_valid_samples(scenario, dist, n_k, 5*n_k)
        samples += samples_k
        labels += [_simulate_and_get_success(scenario, s, **simu_kw)
                   for s in samples_k]
    # Calibrate
    estimator = train_svc(samples, labels, probability=True)
    return estimator
