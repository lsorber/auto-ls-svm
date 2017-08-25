import numpy as np
from scipy.special import gammaln, betaln
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

"""
TODO:
- Large scale implementation (with support for sparse matrices)
- Change kernel radius definition from a multiple k of the nearest neighbour
  distance to the k-nearest neighbour distance.
"""


def robust_normalizer(X, one_sided_extrema=0.05, hypercube_edge_length=1.0):
    """Compute a robust translation and scale parameter."""
    m = np.median(X, axis=0)
    s = np.amax(np.abs(np.percentile(
        X, [one_sided_extrema, 1. - one_sided_extrema], axis=0
    ) - m), axis=0)
    s *= 2. / hypercube_edge_length
    s[s <= np.finfo(X.dtype).eps] = 1.
    return m, s


def ball_volume_loginvdthroot(d):
    """Returns log(V**(-1/d)) where V is the d-volume of a unit ball."""
    return -np.log(np.pi) / 2. + gammaln(d / 2. + 1.) / d


def nearestneighbour_distance_lowerbound(n, d):
    """Returns a lower bound on the expected nearest-neighbour distance of
    n points uniformly distributed over a d-dimensional hypercube."""
    return np.exp(ball_volume_loginvdthroot(d) + betaln((n + 1.) / 2., 1. / d)
                  - np.log(d))


def nearestneighbour_distance(n, d):
    """Returns an estimate of the expected nearest neighbour distance of a row
    in a real-world robustly normalized feature matrix of size n x d."""
    factor = 2. if d > 3 else 1.
    return nearestneighbour_distance_lowerbound(d, n) / factor


def knearestneighbour_distance(X, k=1, max_samples=1000):
    """Estimate the k-nearest neighbour distance."""
    S = X if X.shape[0] < max_samples else \
        X[np.random.choice(X.shape[0], max_samples, replace=False), :]
    X2 = (X ** 2).sum(axis=1)[:, np.newaxis]
    S2 = (S ** 2).sum(axis=1)[:, np.newaxis]
    dist = X2 + (S2.T - 2. * (X @ S.T))
    dist[dist <= np.sqrt(np.finfo(X.dtype).eps)] = np.inf
    dist.sort(axis=0)
    return np.median(np.sqrt(dist[k, :]))


def kernel_radius_to_gamma(kernel_radius, n, d, kernel_value_at_radius=0.5):
    """Converts a kernel radius into a gamma value.

    The kernel radius is defined as a multiple of the estimated nearest
    neighbour distance of a robustly normalized feature matrix of size n x d,
    and is the distance at which the kernel function attains the value
    kernel_value_at_radius.

    Gamma is hyperparameter of the RBF kernel exp(-gamma ||x-y||^2). Finding
    a good value for gamma can be hard to reason about, while setting it in
    terms of the kernel radius as a multiple of the nearest neighbour distance
    should be much more intuitive.

    To compute gamma given the kernel radius, we find:
        exp(-gamma (kernel_radius * nn_dist)^2) = kernel_value_at_radius
        gamma = -log(kernel_value_at_radius) (kernel_radius * nn_dist)^-2
    """
    nn_dist = nearestneighbour_distance(n, d)
    gamma = -np.log(kernel_value_at_radius) / (kernel_radius * nn_dist) ** 2.
    return gamma


class BaseAutoLSSVM(BaseEstimator, RegressorMixin):

    def __init__(self, kernel_radius=0.5, kernel_value_at_radius=0.5, mu=0.5):
        self.kernel_radius = kernel_radius
        self.kernel_value_at_radius = kernel_value_at_radius
        self.mu = mu

    def _normalize_X_y(self, X, y=None):
        """Remove median and scale to that 100*(1 - 2 * one_sided_extrema)%
        of the data is approximately between -0.5 and 0.5."""
        if not hasattr(self, 'X_m_'):
            self.X_m_, self.X_s_ = robust_normalizer(X)
        X = (X - self.X_m_) / self.X_s_
        if y is None:
            return X
        if not hasattr(self, 'y_m_'):
            self.y_m_, self.y_s_ = robust_normalizer(y)
        y = (y - self.y_m_) / self.y_s_
        return X, y

    def fit(self, X, y):
        # Validate input.
        X, y = check_X_y(X, y, accept_sparse=None, dtype='numeric')
        # Normalize input.
        self.n_, self.d_ = X.shape
        X, y = self._normalize_X_y(X, y)
        self.gamma_ = kernel_radius_to_gamma(
            self.kernel_radius, self.n_, self.d_, self.kernel_value_at_radius)
        # Train model.
        self.K_ = pairwise_kernels(
            X, metric='rbf', gamma=self.gamma_, n_jobs=-1)
        return self

    def predict(self, X):
        # Validate input.
        check_is_fitted(self, 'K_')
        X = check_array(X, accept_sparse=None, dtype='numeric')
        # Predict with trained model.
        return self.K_.mean() * np.ones((X.shape[0],))


class AutoLSSVMRegressor(BaseAutoLSSVM):

    def __init__(self, gamma=1.0, eta=1.0):
        super(AutoLSSVMRegressor, self).__init__(gamma=gamma, eta=eta)

    def predict(self, X):
        y = super(AutoLSSVMRegressor, self).predict(X)
        return y
