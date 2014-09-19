"""
Extreme deconvolution solver

This follows Bovy et al.
http://arxiv.org/pdf/0905.2979v2.pdf

Arbitrary mixing matrices R are not yet implemented: currently, this only
works with R = I.
"""
from time import time

import numpy as np
from scipy import linalg

from sklearn.mixture import GMM
from astroML.utils import logsumexp, log_multivariate_gaussian


class XDGMM(object):
    """Extreme Deconvolution

    Fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    n_components: integer
        number of gaussian components to fit to the data
    n_iter: integer (optional)
        number of EM iterations to perform (default=100)
    tol: float (optional)
        stopping criterion for EM iterations (default=1E-5)

    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
    def __init__(self, n_components, n_iter=100, tol=1E-5, verbose=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

        # model parameters: these are set by the fit() method
        self.V = None
        self.mu = None
        self.alpha = None

    def fit(self, X, Xerr, R=None):
        """Fit the XD model to data

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)
        R : array_like
            (TODO: not implemented)
            Transformation matrix from underlying to observed data.  If
            unspecified, then it is assumed to be the identity matrix.
        """
        if R is not None:
            raise NotImplementedError("mixing matrix R is not yet implemented")

        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        # initialize components via a few steps of GMM
        # this doesn't take into account errors, but is a fast first-guess
        gmm = GMM(self.n_components, n_iter=10, covariance_type='full').fit(X)
        self.mu = gmm.means_
        self.alpha = gmm.weights_
        self.V = gmm.covars_

        logL = self.logL(X, Xerr)

        for i in range(self.n_iter):
            t0 = time()
            self._EMstep(X, Xerr)
            logL_next = self.logL(X, Xerr)
            t1 = time()

            if self.verbose:
                print "%i: log(L) = %.5g" % (i + 1, logL_next)
                print "    (%.2g sec)" % (t1 - t0)

            if logL_next < logL + self.tol:
                break
            logL = logL_next

        return self

    def logprob_a(self, X, Xerr):
        """
        Evaluate the probability for a set of points

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)

        Returns
        -------
        p: ndarray
            Probabilities.  shape = (n_samples,)
        """
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        T = Xerr + self.V

        return log_multivariate_gaussian(X, self.mu, T)

    def logL(self, X, Xerr):
        """Compute the log-likelihood of data given the model
        KN: Comparable to the GMM.score(X) function

        Parameters
        ----------
        X: array_like
            data, shape = (n_samples, n_features)
        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)

        Returns
        -------
        logL : float
            log-likelihood
        """
        return np.sum(logsumexp(self.logprob_a(X, Xerr), -1))

    def _EMstep(self, X, Xerr):
        """
        Perform the E-step (eq 16 of Bovy et al)
        """
        n_samples, n_features = X.shape

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]

        w_m = X - self.mu

        T = Xerr + self.V

        #------------------------------------------------------------
        # compute inverse of each covariance matrix T
        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                       n_features, n_features])
        Tinv = np.array([linalg.inv(T[i])
                         for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)

        #------------------------------------------------------------
        # evaluate each mixture at each point
        N = np.exp(log_multivariate_gaussian(X, self.mu, T, Vinv=Tinv))

        #------------------------------------------------------------
        # E-step:
        #  compute q_ij, b_ij, and B_ij
        q = (N * self.alpha) / np.dot(N, self.alpha)[:, None]

        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = self.mu + np.sum(self.V * tmp[:, :, np.newaxis, :], -1)

        tmp = np.sum(Tinv[:, :, :, :, np.newaxis]
                     * self.V[:, np.newaxis, :, :], -2)
        B = self.V - np.sum(self.V[:, :, :, np.newaxis]
                            * tmp[:, :, np.newaxis, :, :], -2)

        #------------------------------------------------------------
        # M-step:
        #  compute alpha, m, V
        qj = q.sum(0)

        self.alpha = qj / n_samples

        self.mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]

        m_b = self.mu - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        self.V = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]

    def sample(self, size=1):
        shape = tuple(np.atleast_1d(size)) + (self.mu.shape[1],)
        npts = np.prod(size)

        alpha_cs = np.cumsum(self.alpha)
        r = np.atleast_1d(np.random.random(size))
        r.sort()

        ind = r.searchsorted(alpha_cs)
        ind = np.concatenate(([0], ind))
        if ind[-1] != size:
            ind[-1] = size

        draw = np.vstack([np.random.multivariate_normal(self.mu[i],
                                                        self.V[i],
                                                        (ind[i + 1] - ind[i],))
                          for i in range(len(self.alpha))])

        return draw.reshape(shape)


# -----------------------------------------------------------------------
    def score_samples(self, X, Xerr):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.mu.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

    #### offending line
    ## have to check if there is the self.weights_ these are the weights of
    ## the different gaussians that are being mixed
    #### original line from scikit learn GMM
    ## check the types of return of log_multivariate_normal_density
    ## - [x] check the types of return of self.alphas - same as gmm.weights_
    ##
    ## lpr = (log_multivariate_normal_density(X, self.means_, self.covars_,
    ##                                       self.covariance_type)
    ## logprob_a computes the log multivariate gaussian for one gaussian at
    ## time and give return of N(x|mu, V) but the algorithm might be smart
    ## enough to
    ## original code computes an array of instead
    ## - need to find the corresponding
    ## mu and V for each components to compute
    ## [N(x|mu_1, V_1), N(x|mu_2, V_2), ..., N(x|mu_n_comp, V_n_comp)]

    ## following line is supposed to add in so each log P of a particular
    ## normal in the mixture add with the log weight of that log

        lpr = (self.logprob_a(X, Xerr) + np.log(self.alpha))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def predict(self, X, Xerr):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob, responsibilities = self.score_samples(X, Xerr)
        return responsibilities.argmax(axis=1)

