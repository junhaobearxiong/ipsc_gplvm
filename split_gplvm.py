import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from typing import Optional, List

from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, triangular,to_default_float
from gpflow.models.model import BayesianModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

import sys


class SplitGPLVM(BayesianModel, InternalDataTrainingLossMixin):

    def __init__(
        self,
        data: OutputData,
        split_space: bool, 
        Xp_mean: tf.Tensor,
        Xp_var: tf.Tensor,
        pi: tf.Tensor,
        kernel_K: List[Kernel],
        Zp: tf.Tensor,
        q_mu=None,
        q_sqrt=None,
        Xs_mean=None,
        Xs_var=None,
        kernel_s=None,
        Zs=None,
        Xs_prior_mean=None,
        Xs_prior_var=None,
        Xp_prior_mean=None,
        Xp_prior_var=None,
        pi_prior=None
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param: split_space, if true, have both shared and private space; 
            if false, only have private spaces (note: to recover GPLVM, set split_space=False and let K=1)
        :param Xp_mean: mean latent positions in the private space [N, Qp] (Qp is the dimension of the private space)
        :param Xp_var: variance of the latent positions in the private space [N, Qp]
        :param pi: mixture responsibility of each category to each point [N, K] (K is the number of categories), i.e. q(c)
        :param kernel_K: private space kernel, one for each category
        :param Zp: inducing inputs of the private space [M, Qp]
        :param num_inducing_variables: number of inducing points, M
        :param q_mu: mean of the inducing variables U [M, D], i.e m in q(U) ~ N(U | m, S)
        :param q_sqrt: cholesky of the covariance matrix of the inducing variables [D, M, M]
        :param Xs_mean: mean latent positions in the shared space [N, Qs] (Qs is the dimension of the shared space). i.e. mus in q(Xs) ~ N(Xs | mus, Ss)
        :param Xs_var: variance of latent positions in shared space [N, Qs], i.e. Ss, assumed diagonal
        :param kernel_s: shared space kernel 
        :param Zs: inducing inputs of the shared space [M, Qs] (M is the number of inducing points)
        :param Xs_prior_mean: prior mean used in KL term of bound, [N, Qs]. By default 0. mean in p(Xs)
        :param Xs_prior_var: prior variance used in KL term of bound, [N, Qs]. By default 1. variance in p(Xs)
        :param Xp_prior_mean: prior mean used in KL term of bound, [N, Qp]. By default 0. mean in p(Xp)
        :param Xp_prior_var: prior variance used in KL term of bound, [N, Qp]. By default 1. variance in p(Xp)
        :param pi_prior: prior mixture weights used in KL term of the bound, [N, K]. By default uniform. p(c)        
        """

        # if don't want shared space, set shared space to none --> get a mixture of GPLVM
        # if don't want private space, set shared space to none, set K = 1 and only include 1 kernel in `kernel_K` --> recover the original GPLVM 

        # TODO: think about how to do this with minibatch
        # it's awkward since w/ minibatch the model usually doesn't store the data internally
        # but for gplvm, you need to keep the q(xn) for all the n's
        # so you need to know which ones to update for each minibatch, probably can be solved but not pretty
        # using inference network / back constraints will solve this, since we will be keeping a global set of parameters
        # rather than a set for each q(xn)
        self.N, self.D = data.shape
        self.Qp = Xp_mean.shape[1]
        self.K = pi.shape[1]
        self.split_space = split_space

        assert Xp_var.ndim == 2
        assert len(kernel_K) == self.K
        assert np.all(Xp_mean.shape == Xp_var.shape)
        assert Xp_mean.shape[0] == self.N, "Xp_mean and Y must be of same size"
        assert pi.shape[0] == self.N, "pi and Y must be of the same size"

        super().__init__()
        self.likelihood = likelihoods.Gaussian()
        self.kernel_K = kernel_K
        self.data = data_input_to_tensor(data)
        # the covariance of q(X) as a [N, Q] matrix, the assumption is that Sn's are diagonal
        # i.e. the latent dimensions are uncorrelated
        # otherwise would require a [N, Q, Q] matrix
        self.Xp_mean = Parameter(Xp_mean)
        self.Xp_var = Parameter(Xp_var, transform=positive())
        self.pi = Parameter(pi, transform=tfp.bijectors.SoftmaxCentered())
        self.Zp = inducingpoint_wrapper(Zp)
        self.M = len(self.Zp)

        # initialize the variational parameters for q(U), same way as in SVGP
        # q_diag is false because natural gradient only works for full covariance
        q_mu = np.zeros((self.M, self.D)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, D]

        if q_sqrt is None:
            q_sqrt = [
                np.eye(self.M, dtype=default_float()) for _ in range(self.D)
            ]
            q_sqrt = np.array(q_sqrt)
            self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [D, M, M]
        else:
            assert q_sqrt.ndim == 3
            assert q_sqrt.shape[0] == self.D
            assert q_sqrt.shape[1] == self.M and q_sqrt.shape[2] == self.M
            self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [D, M, M]

        # deal with parameters for the prior 
        if Xp_prior_mean is None:
            Xp_prior_mean = tf.zeros((self.N, self.Qp), dtype=default_float())
        if Xp_prior_var is None:
            Xp_prior_var = tf.ones((self.N, self.Qp))
        if pi_prior is None:
            pi_prior = tf.ones((self.N, self.K), dtype=default_float()) * 1/self.K

        self.Xp_prior_mean = tf.convert_to_tensor(np.atleast_1d(Xp_prior_mean), dtype=default_float())
        self.Xp_prior_var = tf.convert_to_tensor(np.atleast_1d(Xp_prior_var), dtype=default_float()) 
        self.pi_prior = tf.convert_to_tensor(np.atleast_1d(pi_prior), dtype=default_float()) 


        # if we have both shared space and private space, need to initialize the parameters for the shared space
        if split_space:
            assert Xs_mean is not None and Xs_var is not None and kernel_s is not None and Zs is not None, 'Xs_mean, Xs_var, kernel_s, Zs need to be initialize if `split_space=True`'
            assert Xs_var.ndim == 2 
            assert np.all(Xs_mean.shape == Xs_var.shape)
            assert Xs_mean.shape[0] == self.N, "Xs_mean and Y must be of same size"
            self.Qs = Xs_mean.shape[1]
            self.kernel_s = kernel_s
            self.Xs_mean = Parameter(Xs_mean)
            self.Xs_var = Parameter(Xs_var, transform=positive())
            self.Zs = inducingpoint_wrapper(Zs)

            if len(Zs) != len(Zp):
                raise ValueError(
                    '`Zs` and `Zp` should have the same length'
                )

            if Xs_prior_mean is None:
                Xs_prior_mean = tf.zeros((self.N, self.Qs), dtype=default_float())
            if Xs_prior_var is None:
                Xs_prior_var = tf.ones((self.N, self.Qs))
            self.Xs_prior_mean = tf.convert_to_tensor(np.atleast_1d(Xs_prior_mean), dtype=default_float())
            self.Xs_prior_var = tf.convert_to_tensor(np.atleast_1d(Xs_prior_var), dtype=default_float())



    # KL[q(x) || p(x)] when both q, p are multivariate normals
    @tf.function
    def kl_mvn(self, X_mean, X_var, X_prior_mean, X_prior_var):
        # TODO: right now for the covariance of q(U), we are only considering its diagonal elements
        dX_var = (
            X_var
            if X_var.shape.ndims == 2
            else tf.transpose(tf.linalg.diag_part(X_var))
        )
        NQ = to_default_float(tf.size(X_mean))
        # log of determinant of diagonal matrix = log of product of entries = sum of logs of entries 
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(X_prior_var))
        KL -= 0.5 * NQ
        # KL is additive for independent distribution (sums over N)
        # trace sums over Q (see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)
        KL += 0.5 * tf.reduce_sum(
            (tf.square(X_mean - X_prior_mean) + dX_var) / X_prior_var
        )
        return KL


    # KL[q(c) || p(c)] when both q, p are categoricals 
    @tf.function
    def kl_categorical(self, pi, pi_prior):
        KL = tf.reduce_sum(
            pi * (tf.math.log(pi) - tf.math.log(pi_prior))
        )
        return KL


    @tf.function
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """  

        # defining a sets of vectorized function for usage in `tf.vectorized_map`

        # take the outer product of a pair of rows
        @tf.function
        def row_outer_product(args):
            a, b = args
            a = tf.expand_dims(a, -1)
            b = tf.expand_dims(b, -1)
            return a @ tf.transpose(b)

        # repeat matrix A N times on a newly created first axis 
        # so the new shape is [N, A.shape] 
        @tf.function
        def repeat_N(A):
            return tf.repeat(tf.expand_dims(A, 0), self.N, axis=0)

        @tf.function
        def triang_solve(args):
            L, rhs = args
            return tf.linalg.triangular_solve(L, rhs)

        @tf.function
        def triang_solve_transpose(args):
            L, rhs = args
            return tf.linalg.triangular_solve(tf.transpose(L), rhs, lower=False)

        @tf.function
        def matmul_vectorized(args):
            A, B = args
            return tf.matmul(A, B)

        # [N, D, M, M] --> [N]
        # each term is sum_{d=1}^D Tr[M, M]
        # arg: [D, M, M], needs to be squared
        @tf.function
        def sum_d_trace(arg):
            trace_D = tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(x)), arg)
            return tf.reduce_sum(trace_D)

        # trace of a matrix
        @tf.function
        def trace_tf(A):
            return tf.reduce_sum(tf.linalg.diag_part(A))


        Y = self.data

        # specify qXp, the variational distribution q(X): each x_n is independent w/ N(x_n | \mu_n, S_n)
        # \mu_n \in R^q given by each row of `X_data_mean`
        # S_n \in R^qxq diagonal, so equivalently given by each row of `X_data_var`
        qXp = DiagonalGaussian(self.Xp_mean, self.Xp_var)

        # if split space, specify qXs
        # compute psi statistics for the shared space, keep the original shape of psi statistics, use qXs and kernel_s
        # psi0s is N-vector
        # psi1s is [N, M]
        # psi2s is [N, M, M]
        # also compute the covariance matrix Kuu for the shared space
        if self.split_space:
            qXs = DiagonalGaussian(self.Xs_mean, self.Xs_var)
            psi0s = expectation(qXs, self.kernel_s)
            psi1s = expectation(qXs, (self.kernel_s, self.Zs))
            psi2s = expectation(qXs, (self.kernel_s, self.Zs), (self.kernel_s, self.Zs))
            cov_uu_s = covariances.Kuu(self.Zs, self.kernel_s, jitter=default_jitter())


        # loop over k, for each k use kernel_K[k] and qXp, compute psi0k, psi1k, psi2k, then store the psi statistics for all k together
        # for each k: if no shared space, then psi0[:, k] = psi0k, psi1[:, :, k] = psi1k, psi2[:, :, :, k] = psi2k
        # if have shared space, then psi0[:, k] = psi0s + psi0k, psi1[:, :, k] = psi1s + psi1k
        # psi2[:, :, :, k] = psi2s + psi2k (the cross terms are added later)
        # then, for each n, psi2[n, :, :, k] = psi1s[n, :]^T dot psi1k[n, :] + psi1k[n, :]^T dot psi1s[n, :] (both are [M, M])
        # psi0 is [N, K] so psi0[n, k] gives a real value
        # psi1 is [N, M, K], so psi1[n, :, k] gives us a M-vector
        # psi2 is [N, M, M, K], so psi2[n, :, :, k] gives us a [M x M] matrix
        psi0 = []
        psi1 = []
        psi2 = []
        for k, kernel_k in enumerate(self.kernel_K):
            psi0k = expectation(qXp, kernel_k)
            psi1k = expectation(qXp, (kernel_k, self.Zp))
            psi2k = expectation(qXp, (kernel_k, self.Zp), (kernel_k, self.Zp))
            if self.split_space:
                psi0.append(psi0s + psi0k)            
                psi1.append(psi1s + psi1k)
                # add the cross-covariance terms, require computation separately for each n
                sxk = tf.vectorized_map(row_outer_product, (psi1s, psi1k))
                kxs = tf.vectorized_map(row_outer_product, (psi1k, psi1s))
                psi2.append(psi2s + psi2k + sxk + kxs)
            else:
                psi0.append(psi0k)
                psi1.append(psi1k)
                psi2.append(psi2k)
        psi0 = tf.stack(psi0, axis=-1)
        psi1 = tf.stack(psi1, axis=-1)
        psi2 = tf.stack(psi2, axis=-1)

        
        # make K cov_uu_k using Zp and kernel_k
        # K cholesky, repeat N times for later use
        # L is [N x M x M x K]
        # these are the Kuu matrices
        L = []
        for k, kernel_k in enumerate(self.kernel_K):
            cov_uu_k = covariances.Kuu(self.Zp, kernel_k, jitter=default_jitter())
            if self.split_space:
                L.append(tf.linalg.cholesky(cov_uu_s + cov_uu_k))
            else:
                L.append(tf.linalg.cholesky(cov_uu_k))
        L = tf.stack(L, axis=-1)
        L = repeat_N(L)
        sigma2 = self.likelihood.variance

        # use `tf.vectorized_map` to avoid writing a loop over N, but it requires every matrix to have N on axis 0
        # so we need to repeat certain matrices that are the same for all N (e.g. L)
        # note we can use `tf.vectorized_map` because the computations are decomposable for each n,
        # i.e. they can be computed in any order over n
        Fq = []
        Yn2 = tf.reduce_sum(tf.square(Y), axis=1)
        for k in range(self.K):
            # compute intermediate matrices for easier computation involving \inv{Kuu}
            tmp = tf.vectorized_map(triang_solve, (L[..., k], psi2[..., k])) # [N, M, M]
            A = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp)) # \inv{Kuu} * Psi2: [N, M, M]
            
            #pos_def = tf.vectorized_map(lambda x: is_pos_def(x), psi2[..., k])
            #print(np.all(pos_def))
            # psi2 is not produced using w/ `covariances.Kuu`, but it should still be PD
            # we should add jitter before doing cholesky
            jitter_mtx = default_jitter() * tf.eye(self.M, dtype=default_float())
            LB = tf.vectorized_map(lambda x: tf.linalg.cholesky(x + jitter_mtx), psi2[..., k]) # [N, M, M]   
            tmp1 = tf.vectorized_map(triang_solve, (L[..., k], LB)) # [N, M, M]
            C = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp1)) # sqrt(\inv{Kuu} * Psi2 * \inv{Kuu}): [N, M, M]

            D = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_mu)), C)) # sqrt(M^T * \inv{Kuu} * Psi2 * \inv{Kuu} * M): [N, M, D]

            tmp2 = tf.vectorized_map(triang_solve, (L[..., k], repeat_N(self.q_mu)))
            E = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp2)) # \inv{Kuu} * M: [N, M, D]

            # q_sqrt is already the cholesky
            F = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_sqrt, perm=[0, 2, 1])), C)) # sqrt(S * \inv{Kuu} * Psi2 * \inv{Kuu}): [N, D, M, M]

            tmp3 = tf.vectorized_map(row_outer_product, (Y, psi1[..., k])) # Y^T * Psi1: [N, D, M]
            G = tf.vectorized_map(matmul_vectorized, (tmp3, E)) # Y^T * Psi1 * \inv{Kuu} * M: [N, D, D]

            # compute the lower bound
            # each term added here is length-N vector, each entry representing \sum_{d=1}^D Fdnk for a particular n, k
            Fnk = -0.5 * Yn2 / sigma2
            Fnk += tf.vectorized_map(lambda x: trace_tf(x), G) / sigma2
            Fnk += -0.5 * tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(x)), D) / sigma2
            Fnk += -0.5 * self.D * tf.vectorized_map(lambda x: trace_tf(x), A)  / sigma2 
            Fnk += -0.5 * tf.vectorized_map(lambda x: sum_d_trace(x), F) / sigma2

            Fq.append(Fnk)

        Fq = tf.stack(Fq, axis=-1) # [N, K]
        # psi0 is already [N, K]
        Fq += -0.5 * self.D * psi0 / sigma2
        Fq += -0.5 * self.D * tf.math.log(2 * np.pi * sigma2)
        # weight each entry by the mixture responsibility, then sum over N, K
        bound = tf.reduce_sum(Fq * self.pi)

        # compute KL 
        KL_p = self.kl_mvn(self.Xp_mean, self.Xp_var, self.Xp_prior_mean, self.Xp_prior_var)
        # assumes p(U) has identity covariance for each d
        # simplifies since otherwise we need to compute Kuu, so need to split this KL for different k
        KL_u = gauss_kl(self.q_mu, self.q_sqrt)
        KL_c = self.kl_categorical(self.pi, self.pi_prior)
        bound += - KL_p - KL_u - KL_c
        if self.split_space:
            KL_s = self.kl_mvn(self.Xs_mean, self.Xs_var, self.Xs_prior_mean, self.Xs_prior_var)
            bound += - KL_s
        
        return bound
        

    @tf.function
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()