import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from typing import Optional, List

from gpflow import covariances, kernels, likelihoods, kullback_leiblers
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
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
        # q_mu: List[K], mean of the inducing variables U [M, D], i.e m in q(U) ~ N(U | m, S), 
        #   initialized as zeros
        # q_sqrt: List[K], cholesky of the covariance matrix of the inducing variables [D, M, M]
        #   q_diag is false because natural gradient only works for full covariance
        #   initialized as all identities
        # we need K sets of q(Uk), each approximating fs+fk
        self.q_mu = []
        self.q_sqrt = []
        for k in range(self.K):
            q_mu = np.zeros((self.M, self.D))
            q_mu = Parameter(q_mu, dtype=default_float())  # [M, D]
            self.q_mu.append(q_mu)

            q_sqrt = [
                np.eye(self.M, dtype=default_float()) for _ in range(self.D)
            ]
            q_sqrt = np.array(q_sqrt)
            q_sqrt = Parameter(q_sqrt, transform=triangular())  # [D, M, M]
            self.q_sqrt.append(q_sqrt)

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

        self.Fq = tf.zeros((self.N, self.K), dtype=default_float())


    # KL[q(x) || p(x)] when both q, p are multivariate normals
    @tf.function
    def kl_mvn(self, X_mean, X_var, X_prior_mean, X_prior_var):
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


    #@tf.function
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


        # self.pred_Y = []

        # use `tf.vectorized_map` to avoid writing a loop over N, but it requires every matrix to have N on axis 0
        # so we need to repeat certain matrices that are the same for all N (e.g. L)
        # note we can use `tf.vectorized_map` because the computations are decomposable for each n,
        # i.e. they can be computed in any order over n
        Fq = []
        Yn2 = tf.reduce_sum(tf.square(Y), axis=1)
        for k in range(self.K):
            # compute intermediate matrices for easier computation involving \inv{Kuu}
            # A is the same as AAT in gplvm, transposing L is the correct thing to do
            # but the two end up being the same since we only care about the trace
            tmp = tf.vectorized_map(triang_solve, (L[..., k], psi2[..., k])) # [N, M, M]
            A = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp)) # \inv{Kuu} * Psi2: [N, M, M]

            #pos_def = tf.vectorized_map(lambda x: is_pos_def(x), psi2[..., k])
            #print(np.all(pos_def))
            # psi2 is not produced using w/ `covariances.Kuu`, but it should still be PD
            # we should add jitter before doing cholesky
            #jitter_mtx = default_jitter() * tf.eye(self.M, dtype=default_float())
            jitter_mtx = 1e-10 * tf.eye(self.M, dtype=default_float())
            LB = tf.vectorized_map(lambda x: tf.linalg.cholesky(x + jitter_mtx), psi2[..., k]) # [N, M, M]  
            tmp1 = tf.vectorized_map(triang_solve, (L[..., k], LB)) # [N, M, M]
            C = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp1)) # sqrt(\inv{Kuu} * Psi2 * \inv{Kuu}): [N, M, M]

            D = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_mu[k])), C)) # sqrt(M^T * \inv{Kuu} * Psi2 * \inv{Kuu} * M): [N, D, M]

            tmp2 = tf.vectorized_map(triang_solve, (L[..., k], repeat_N(self.q_mu[k])))
            E = tf.vectorized_map(triang_solve_transpose, (L[..., k], tmp2)) # \inv{Kuu} * M: [N, M, D]

            # q_sqrt is already the cholesky
            F = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_sqrt[k], perm=[0, 2, 1])), C)) # sqrt(S * \inv{Kuu} * Psi2 * \inv{Kuu}): [N, D, M, M]

            tmp3 = tf.vectorized_map(row_outer_product, (Y, psi1[..., k])) # Y^T * Psi1: [N, D, M]
            G = tf.vectorized_map(matmul_vectorized, (tmp3, E)) # Y^T * Psi1 * \inv{Kuu} * M: [N, D, D]

            # for debugging 
            # self.pred_Y.append(tf.reshape(tf.vectorized_map(matmul_vectorized, (tf.expand_dims(psi1[..., k], 1), E)), (self.N, self.D))) # Psi1 * \inv{Kuu} * M: [N, D]

            # compute the lower bound
            # each term added here is length-N vector, each entry representing \sum_{d=1}^D Fdnk for a particular n, k
            Fnk = -0.5 * Yn2 / sigma2
            Fnk += tf.vectorized_map(lambda x: trace_tf(x), G) / sigma2
            Fnk += -0.5 * tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(x)), D) / sigma2
            Fnk += 0.5 * self.D * tf.vectorized_map(lambda x: trace_tf(x), A)  / sigma2 
            Fnk += -0.5 * tf.vectorized_map(lambda x: sum_d_trace(x), F) / sigma2

            Fq.append(Fnk)

        Fq = tf.stack(Fq, axis=-1) # [N, K]
        # psi0 is already [N, K]
        Fq += -0.5 * self.D * psi0 / sigma2
        Fq += -0.5 * self.D * tf.math.log(2 * np.pi * sigma2)

        # for debugging 
        #self.Fq = Fq
        # self.pred_Y = tf.stack(self.pred_Y, axis=-1) # [N, D, K]

        # weight each entry by the mixture responsibility, then sum over N, K
        bound = tf.reduce_sum(Fq * self.pi)

        # compute KL 
        KL_p = self.kl_mvn(self.Xp_mean, self.Xp_var, self.Xp_prior_mean, self.Xp_prior_var)
        KL_c = self.kl_categorical(self.pi, self.pi_prior)
        KL_u = 0
        prior_Kuu = np.zeros((self.M, self.M))
        if self.split_space:
            KL_s = self.kl_mvn(self.Xs_mean, self.Xs_var, self.Xs_prior_mean, self.Xs_prior_var)
            bound += - KL_s
            prior_Kuu += covariances.Kuu(self.Zs, self.kernel_s, jitter=default_jitter())
        for k in range(self.K):
            prior_Kuu_k = covariances.Kuu(self.Zp, self.kernel_K[k], jitter=default_jitter())
            KL_u += kullback_leiblers.gauss_kl(q_mu=self.q_mu[k], q_sqrt=self.q_sqrt[k], K=prior_Kuu+prior_Kuu_k)
        bound += - KL_p - KL_u - KL_c

        return bound
        

    @tf.function
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

# Split GPLVM with a separate approximation for fs and fk
# assumes split_space = True, otherwise this is equivalent to SplitGPLVM
class SplitGPLVMApprox(SplitGPLVM):
    def __init__(
        self,
        data: OutputData,
        Xp_mean: tf.Tensor,
        Xp_var: tf.Tensor,
        pi: tf.Tensor,
        kernel_K: List[Kernel],
        Zp: tf.Tensor,
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
        super().__init__(
            data=data,
            split_space=True, 
            Xp_mean=Xp_mean,
            Xp_var=Xp_var,
            pi=pi,
            kernel_K=kernel_K,
            Zp=Zp,
            Xs_mean=Xs_mean,
            Xs_var=Xs_var,
            kernel_s=kernel_s,
            Zs=Zs,
            Xs_prior_mean=Xs_prior_mean,
            Xs_prior_var=Xs_prior_var,
            Xp_prior_mean=Xp_prior_mean,
            Xp_prior_var=Xp_prior_var,
            pi_prior=pi_prior
        )
        # q(Us | Ms, Ss)
        q_mu = np.zeros((self.M, self.D))
        self.q_mu_s = Parameter(q_mu, dtype=default_float())  # [M, D]

        q_sqrt = [
            np.eye(self.M, dtype=default_float()) for _ in range(self.D)
        ]
        q_sqrt = np.array(q_sqrt)
        self.q_sqrt_s = Parameter(q_sqrt, transform=triangular())  # [D, M, M]


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

        qXs = DiagonalGaussian(self.Xs_mean, self.Xs_var)
        psi0s = expectation(qXs, self.kernel_s)
        psi1s = expectation(qXs, (self.kernel_s, self.Zs))
        psi2s = expectation(qXs, (self.kernel_s, self.Zs), (self.kernel_s, self.Zs))
        cov_uu_s = covariances.Kuu(self.Zs, self.kernel_s, jitter=default_jitter())
        Ls = tf.linalg.cholesky(cov_uu_s)
        Ls = repeat_N(Ls) # [N x M x M]

        # loop over k, for each k use kernel_K[k] and qXp, compute psi0k, psi1k, psi2k, then store the psi statistics for all k together
        # for each k: if no shared space, then psi0[:, k] = psi0k, psi1[:, :, k] = psi1k, psi2[:, :, :, k] = psi2k
        # if have shared space, then psi0[:, k] = psi0s + psi0k, psi1[:, :, k] = psi1s + psi1k
        # psi2[:, :, :, k] = psi2s + psi2k (the cross terms are added later)
        # then, for each n, psi2[n, :, :, k] = psi1s[n, :]^T dot psi1k[n, :] + psi1k[n, :]^T dot psi1s[n, :] (both are [M, M])
        # psi0 is [N, K] so psi0[n, k] gives a real value
        # psi1 is [N, M, K], so psi1[n, :, k] gives us a M-vector
        # psi2 is [N, M, M, K], so psi2[n, :, :, k] gives us a [M x M] matrix
        qXp = DiagonalGaussian(self.Xp_mean, self.Xp_var)
        psi0k = []
        psi1k = []
        psi2k = []
        psi2ks = []
        psi2sk = []
        for k, kernel_k in enumerate(self.kernel_K):
            psi0 = expectation(qXp, kernel_k)
            psi1 = expectation(qXp, (kernel_k, self.Zp))
            psi2 = expectation(qXp, (kernel_k, self.Zp), (kernel_k, self.Zp))
            psi0k.append(psi0)            
            psi1k.append(psi1)
            psi2k.append(psi2)
            # add the cross-covariance terms, require computation separately for each n
            psi2sk.append(tf.vectorized_map(row_outer_product, (psi1s, psi1)))
            #psi2ks.append(tf.vectorized_map(row_outer_product, (psi1, psi1s)))
        psi0k = tf.stack(psi0k, axis=-1)
        psi1k = tf.stack(psi1k, axis=-1)
        psi2k = tf.stack(psi2k, axis=-1)
        psi2sk = tf.stack(psi2sk, axis=-1)
        #psi2ks = tf.stack(psi2ks, axis=-1)  

        # make K cov_uu_k using Zp and kernel_k
        # K cholesky, repeat N times for later use
        # L is [N x M x M x K]
        # these are the Kuu matrices
        Lk = []
        for k, kernel_k in enumerate(self.kernel_K):
            cov_uu_k = covariances.Kuu(self.Zp, kernel_k, jitter=default_jitter())
            Lk.append(tf.linalg.cholesky(cov_uu_k))
        Lk = tf.stack(Lk, axis=-1)
        Lk = repeat_N(Lk)
        
        sigma2 = self.likelihood.variance
        jitter_mtx = 1e-10 * tf.eye(self.M, dtype=default_float())

        tmp = tf.vectorized_map(triang_solve, (Ls, psi2s))
        As = tf.vectorized_map(triang_solve_transpose, (Ls, tmp)) # \inv{Kuu^s} * Psi2s: [N, M, M]

        LBs = tf.vectorized_map(lambda x: tf.linalg.cholesky(x + jitter_mtx), psi2s) # [N, M, M]  
        tmp1 = tf.vectorized_map(triang_solve, (Ls, LBs)) # [N, M, M]
        Cs = tf.vectorized_map(triang_solve_transpose, (Ls, tmp1)) # sqrt(\inv{Kuu^s} * Psi2s * \inv{Kuu^s}): [N, M, M]
        Ds = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_mu_s)), Cs)) # sqrt(Ms^T * \inv{Kuu^s} * Psi2s * \inv{Kuu^s} * Ms): [N, D, M]

        Fs = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_sqrt_s, perm=[0, 2, 1])), Cs)) # sqrt(Ss * \inv{Kuu^s} * Psi2s * \inv{Kuu^s}): [N, D, M, M]

        tmp2 = tf.vectorized_map(triang_solve, (Ls, repeat_N(self.q_mu_s)))
        Es = tf.vectorized_map(triang_solve_transpose, (Ls, tmp2)) # \inv{Kuu^s} * Ms: [N, M, D]
        tmp3 = tf.vectorized_map(row_outer_product, (Y, psi1s)) # Y^T * Psi1: [N, D, M]
        Gs = tf.vectorized_map(matmul_vectorized, (tmp3, Es)) # Y^T * Psi1s * \inv{Kuu^s} * Ms: [N, D, D]

        Fq = []
        Yn2 = tf.reduce_sum(tf.square(Y), axis=1)
        for k in range(self.K):
            tmp = tf.vectorized_map(triang_solve, (Lk[..., k], psi2k[..., k])) # [N, M, M]
            Ak = tf.vectorized_map(triang_solve_transpose, (Lk[..., k], tmp)) # \inv{Kuu^k} * Psi2k: [N, M, M]

            LBk = tf.vectorized_map(lambda x: tf.linalg.cholesky(x + jitter_mtx), psi2k[..., k]) # [N, M, M]  
            tmp1k = tf.vectorized_map(triang_solve, (Lk[..., k], LBk)) # [N, M, M]
            Ck = tf.vectorized_map(triang_solve_transpose, (Lk[..., k], tmp1k)) # sqrt(\inv{Kuu^k} * Psi2k * \inv{Kuu^k}): [N, M, M]
            Dk = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_mu[k])), Ck)) # sqrt(Mk^T * \inv{Kuu^k} * Psi2k * \inv{Kuu^k} * Mk): [N, D, M]

            # q_sqrt is already the cholesky
            Fk = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_sqrt[k], perm=[0, 2, 1])), Ck)) # sqrt(Sk * \inv{Kuu^k} * Psi2k * \inv{Kuu^k}): [N, D, M, M]

            tmp2 = tf.vectorized_map(triang_solve, (Lk[..., k], repeat_N(self.q_mu[k])))
            Ek = tf.vectorized_map(triang_solve_transpose, (Lk[..., k], tmp2)) # \inv{Kuu^k} * Mk: [N, M, D]
            tmp3 = tf.vectorized_map(row_outer_product, (Y, psi1k[..., k])) # Y^T * Psi1k: [N, D, M]
            Gk = tf.vectorized_map(matmul_vectorized, (tmp3, Ek)) # Y^T * Psi1k * \inv{Kuu^k} * Mk: [N, D, D]

            # compute the cross terms 
            tmp1sk = tf.vectorized_map(triang_solve, (Ls, psi2sk[..., k]))
            tmp2sk = tf.vectorized_map(triang_solve_transpose, (Ls, tmp1sk)) # \inv{Kuu^s} * Psi2sk: [N, M, M]
            tmp3sk = tf.vectorized_map(matmul_vectorized, (tmp2sk, Ek)) # \inv{Kuu^s} * Psi2sk * \inv{Kuu^k} * Mk: [N, M, D]
            Dsk = tf.vectorized_map(matmul_vectorized, (repeat_N(tf.transpose(self.q_mu_s)), tmp3sk)) # Ms^T * \inv{Kuu^s} * Psi2sk * \inv{Kuu^k} * Mk: [N, D, D]

            # compute the lower bound
            # each term added here is length-N vector, each entry representing \sum_{d=1}^D Fdnk for a particular n, k
            Fnk = -0.5 * Yn2 / sigma2
            Fnk += tf.vectorized_map(trace_tf, Gs + Gk) / sigma2
            Fnk += -0.5 * tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(x)), Ds) / sigma2
            Fnk += -0.5 * tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(x)), Dk) / sigma2
            # the sum of trace of the 2 cross terms is 2 times the trace of one since they are transpose of one another
            Fnk += - tf.vectorized_map(trace_tf, Dsk) / sigma2 
            Fnk += 0.5 * self.D * tf.vectorized_map(trace_tf, As + Ak)  / sigma2 
            Fnk += -0.5 * tf.vectorized_map(sum_d_trace, Fs) / sigma2
            Fnk += -0.5 * tf.vectorized_map(sum_d_trace, Fk) / sigma2

            Fq.append(Fnk)

        Fq = tf.stack(Fq, axis=-1) # [N, K]
        # psi0 is already [N, K]
        Fq += -0.5 * self.D * (tf.repeat(tf.expand_dims(psi0s, -1), self.K, axis=1) + psi0k) / sigma2
        Fq += -0.5 * self.D * tf.math.log(2 * np.pi * sigma2)

        # weight each entry by the mixture responsibility, then sum over N, K
        bound = tf.reduce_sum(Fq * self.pi)

        # compute KL 
        KL_p = self.kl_mvn(self.Xp_mean, self.Xp_var, self.Xp_prior_mean, self.Xp_prior_var)
        KL_c = self.kl_categorical(self.pi, self.pi_prior)
        KL_s = self.kl_mvn(self.Xs_mean, self.Xs_var, self.Xs_prior_mean, self.Xs_prior_var)
        
        prior_Kuu_s = covariances.Kuu(self.Zs, self.kernel_s, jitter=default_jitter())
        KL_us = kullback_leiblers.gauss_kl(q_mu=self.q_mu_s, q_sqrt=self.q_sqrt_s, K=prior_Kuu_s)
        KL_uk = 0
        for k in range(self.K):
            prior_Kuu_k = covariances.Kuu(self.Zp, self.kernel_K[k], jitter=default_jitter())
            KL_uk += kullback_leiblers.gauss_kl(q_mu=self.q_mu[k], q_sqrt=self.q_sqrt[k], K=prior_Kuu_k)
        bound += - KL_s - KL_p - KL_us - KL_uk - KL_c

        return bound

    @tf.function
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()