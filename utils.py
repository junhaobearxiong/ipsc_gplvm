import numpy as np
import gpflow
import tensorflow as tf
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, List
import time

from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.models import gplvm
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable

from split_gplvm import SplitGPLVM, SplitGPLVMApprox
from test_gplvm import TestGPLVM


'''
    Simulations
'''
def breakpoint_linear(x, ts, k1, k2, c1, sigma):
    '''Function representing a step-wise linear curve with one
    breakpoint located at ts.
    '''
    func = np.piecewise(x, [x < ts], [lambda x: k1 * x + c1,
                                      lambda x: k2 * x + (k1 - k2) * ts + c1])
    return func + np.random.normal(func, sigma)


# X ([num_data, 2]), where X[:,0] is "time", X[:,1] is the values of two branching processes 
def branch_simulation(x, break_pt, k1, k21, k22, c1, sigma):
    func1 = breakpoint_linear(x, break_pt, k1, k21, c1, sigma)
    func2 = breakpoint_linear(x, break_pt, k1, k22, c1, sigma)
    data1 = np.stack([x, func1], axis=1)
    data2 = np.stack([x, func2], axis=1)
    X = np.concatenate([data1, data2], axis=0)
    return X #(X - X.mean()) / X.std()


# generalized the bgp kernel in the following ways:
# 1. all points before branch point xp are constrained to be the same for the 2 GPs
# 2. X is [N, D] D >= 1, and Y is [N, dim], dim >= 1
# 3. Allow constraining based on only the first dimension of X
#   i.e. f(x) = g(x) for all x where x[0] < xp[0]
def branch_kernel(X, xp, dim, x1_only=True):
    num_data = X.shape[0]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0])
    Kff = kernel(X)
    if x1_only:
        mask = (X < xp)[:, 0]
    else:
        mask = np.all(X < xp, axis=1)
    Xs = X[mask, :]
    Kxu = kernel(X, Xs)
    Kuu = kernel(Xs)
    jitter_mtx = 1e-6 * np.eye(Xs.shape[0])
    L = tf.linalg.cholesky(Kuu + jitter_mtx)
    tmp = tf.linalg.triangular_solve(L, tf.transpose(Kxu))
    Kfg = tf.transpose(tmp) @ tmp

    tmp1 = tf.concat([Kff, Kfg], axis=0)
    tmp2 = tf.concat([Kfg, Kff], axis=0)
    Sigma = tf.concat([tmp1, tmp2], axis=1)

    y = np.random.default_rng().multivariate_normal(np.zeros(num_data * 2), Sigma, dim).T
    fx = y[:num_data]
    gx = y[num_data:]
    
    return (fx, gx)


def gen_sim(num_data, dim):
    np.random.seed(1)

    xmin = -5
    xmax = 5
    break_pt = xmin + (xmax - xmin) * 0.5
    k1 = 0.0
    k21 = 0.5
    k22 = -0.5
    c1 = 0
    sigma = 0

    x = np.linspace(xmin, xmax, int(num_data/2), dtype=default_float())
    X = branch_simulation(x, break_pt, k1, k21, k22, c1, sigma)
    labels = np.repeat([0, 1], int(num_data)/2)

    fx, gx = branch_kernel(X, break_pt, dim)

    break_num = x[x < break_pt].size
    halfpt = int(num_data / 2)
    tmp1 = fx[:break_num]
    tmp2 = fx[break_num:halfpt]
    tmp3 = gx[-(halfpt-break_num):]
    Y = np.concatenate([tmp1, tmp2, tmp1, tmp3], axis=0)

    return (X, Y, labels)

'''
    Model initialization
'''
def init_gplvm(Y, latent_dim, kernel=None, num_inducing=None, inducing_variable=None, X_mean_init=None, X_var_init=None):
    num_data = Y.shape[0]  # number of data points

    if X_mean_init is None:
        X_mean_init = tf.constant(PCA(n_components=latent_dim).fit_transform(Y), dtype=default_float())
    else:
        X_mean_init = tf.constant(X_mean_init, dtype=default_float())
    if X_var_init is None:
        X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())
    else:
        X_var_init = tf.constant(X_var_init, dtype=default_float())

    if kernel is None:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*latent_dim)

    if (inducing_variable is None) == (num_inducing is None):
        raise ValueError(
            "GPLVM needs exactly one of `inducing_variable` and `num_inducing`"
        )

    if inducing_variable is None:
        inducing_variable = tf.convert_to_tensor(np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float())
        #inducing_variable = tf.convert_to_tensor(np.linspace(np.min(X_mean_init, axis=0), np.max(X_mean_init, axis=0), num_inducing), dtype=default_float())

    model = gpflow.models.BayesianGPLVM(
        Y,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        inducing_variable=inducing_variable,
    )
    return model


def init_test_gplvm(Y, latent_dim, kernel, num_inducing=None, inducing_variable=None, X_mean_init=None, X_var_init=None):
    num_data = Y.shape[0]  # number of data points

    if X_mean_init is None:
        X_mean_init = tf.constant(PCA(n_components=latent_dim).fit_transform(Y), dtype=default_float())
    else:
        X_mean_init = tf.constant(X_mean_init, dtype=default_float())
    if X_var_init is None:
        X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())
    else:
        X_var_init = tf.constant(X_var_init, dtype=default_float())

    if (inducing_variable is None) == (num_inducing is None):
        raise ValueError(
            "GPLVM needs exactly one of `inducing_variable` and `num_inducing`"
        )

    if inducing_variable is None:
        inducing_variable = tf.convert_to_tensor(np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float())
        #inducing_variable = tf.convert_to_tensor(np.linspace(np.min(X_mean_init, axis=0), np.max(X_mean_init, axis=0), num_inducing), dtype=default_float())

    model = TestGPLVM(
        Y,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        inducing_variable=inducing_variable,
    )
    return model


def init_split_gplvm(Y, split_space, Qp, K, approx=False, kernel_K=None, M=None, Xp_mean_init=None, Xp_var_init=None,
    Zp=None, Zs=None, pi_init=None, Qs=None, Xs_mean_init=None, Xs_var_init=None,
    Xs_prior_var=None, Xp_prior_var=None):
    N = Y.shape[0]  # number of data points
    if split_space:
        assert Qs is not None, 'if split_space, need to speficy Qs'
        Q = Qs + Qp
    else:
        Q = Qp

    X_pca = PCA(n_components=Q).fit_transform(Y)

    if Xp_mean_init is None:
        Xp_mean_init = tf.constant(X_pca[:, :Qp], dtype=default_float())
    if Xp_var_init is None:
        Xp_var_init = tf.ones((N, Qp), dtype=default_float())

    if kernel_K is None:
        kernel_K = [gpflow.kernels.SquaredExponential(lengthscales=tf.convert_to_tensor([1.0] * Qp, dtype=default_float())) for k in range(K)]

    if (Zp is None) == (M is None):
        raise ValueError(
            "GPLVM needs exactly one of `Zp` and `M`"
        )

    if Zp is None:
        Zp = tf.convert_to_tensor(np.random.permutation(Xp_mean_init.numpy())[:M], dtype=default_float())

    if pi_init is None:
        pi_init=tf.constant(np.random.dirichlet(alpha=[2 for _ in range(K)], size=(N)), dtype=default_float())

    if split_space:
        if Xs_mean_init is None:
            Xs_mean_init = tf.constant(X_pca[:, :Qs], dtype=default_float())
        if Xs_var_init is None:
            Xs_var_init = tf.ones((N, Qs), dtype=default_float())
        if Zs is None:
            Zs = tf.convert_to_tensor(np.random.permutation(Xs_mean_init.numpy())[:M], dtype=default_float())
        kernel_s = gpflow.kernels.SquaredExponential(lengthscales=tf.convert_to_tensor([1.0] * Qs, dtype=default_float()))
        model = SplitGPLVM(
            data=Y,
            split_space=True,
            Xp_mean=Xp_mean_init,
            Xp_var=Xp_var_init,
            pi=pi_init,
            kernel_K=kernel_K,
            Zp=Zp,
            Xs_mean=Xs_mean_init,
            Xs_var=Xs_var_init,
            kernel_s=kernel_s,
            Zs=Zs,
            Xp_prior_var=Xp_prior_var,
            Xs_prior_var=Xs_prior_var
        )
        if approx:
            model = SplitGPLVMApprox(
                data=Y,
                Xp_mean=Xp_mean_init,
                Xp_var=Xp_var_init,
                pi=pi_init,
                kernel_K=kernel_K,
                Zp=Zp,
                Xs_mean=Xs_mean_init,
                Xs_var=Xs_var_init,
                kernel_s=kernel_s,
                Zs=Zs,
                Xp_prior_var=Xp_prior_var,
                Xs_prior_var=Xs_prior_var
            )
    else:
        model = SplitGPLVM(
            data=Y,
            split_space=False,
            Xp_mean=Xp_mean_init,
            Xp_var=Xp_var_init,
            pi=pi_init,
            kernel_K=kernel_K,
            Zp=Zp,
            Xp_prior_var=Xp_prior_var
        )
    
    return model


'''
    Optimization
'''
def train_scipy(m, maxiter=2000, step=True):
    log_elbo = []
    # log_pi = []

    def step_callback(step, variables, values):
        elbo = m.elbo()
        print('step {} elbo: {}'.format(step, elbo))
        log_elbo.append(elbo)
        # log_pi.append(m.pi.numpy())

    opt = gpflow.optimizers.Scipy()
    if step:
        _ = opt.minimize(
            m.training_loss,
            method="BFGS",
            variables=m.trainable_variables,
            options=dict(maxiter=ci_niter(maxiter), disp=True),
            step_callback=step_callback,
            compile=True
        )
    else:
       _ = opt.minimize(
            m.training_loss,
            method="BFGS",
            variables=m.trainable_variables,
            options=dict(maxiter=ci_niter(maxiter), disp=True),
            compile=True
        ) 
    # return (log_elbo, log_pi)
    return log_elbo
        

def train_natgrad_adam(model, approx=False, num_iterations=2000, log_freq=10):
    
    natgrad_opt = NaturalGradient(gamma=1.0)
    adam_opt = tf.optimizers.Adam(learning_rate=0.01)
    variational_params = list(zip(model.q_mu, model.q_sqrt))
    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)
    if approx:
        variational_params.append((model.q_mu_s, model.q_sqrt_s))
        gpflow.set_trainable(model.q_mu_s, False)
        gpflow.set_trainable(model.q_sqrt_s, False)

    @tf.function
    def optimization_step():
        natgrad_opt.minimize(model.training_loss, var_list=variational_params)
        adam_opt.minimize(model.training_loss, var_list=model.trainable_variables)
        #return (model.elbo(), model.Fq)
        return model.elbo()

    log_elbo = []
    #log_Fq = []
    # log_predY = []
    tol = 1e-4

    print('initial elbo {:.4f}'.format(model.elbo()))

    for step in range(num_iterations):
        start_time = time.time()
        #elbo, Fq = optimization_step()
        elbo = optimization_step()
        log_elbo.append(elbo)
        #log_Fq.append(Fq.numpy())
        # log_predY.append(pred_Y.numpy())

        if step > 0 and np.abs(elbo - log_elbo[-2]) < tol:
            print('converge at iteration {} elbo {:.4f}'.format(step+1, elbo))
            break
        if (step + 1)  % log_freq == 0:
            print('iteration {} elbo {:.4f}, took {:.4f}s'.format(step+1, elbo, time.time()-start_time))
            
    #return (log_elbo, log_Fq)
    return log_elbo


def train_adam(model, num_iterations=2000, log_freq=10):
    adam_opt = tf.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def optimization_step():
        adam_opt.minimize(model.training_loss, var_list=model.trainable_variables)
        return model.elbo()

    log_elbo = []
    tol = 1e-4

    print('initial elbo {:.4f}'.format(model.elbo()))

    for step in range(num_iterations):
        start_time = time.time()
        elbo = optimization_step()
        log_elbo.append(elbo)
        if step > 0 and np.abs(elbo - log_elbo[-2]) < tol:
            print('converge at iteration {} elbo {:.4f}'.format(step+1, elbo))
            break
        if (step + 1)  % log_freq == 0:
            print('iteration {} elbo {:.4f}, took {:.4f}s'.format(step+1, elbo, time.time()-start_time))
    return log_elbo

'''
    Debugging
'''
# first compute the predicted observation by each of the K mixture: [N, D, K]
# then weight each mixture prediction by the mixture weight
# resulting in [N, D]
def get_pred_Y(m, by_K=False):
    pred_Y = np.zeros((m.N, m.D))

    if by_K:
        pred_Y_k = np.zeros((m.N, m.D, m.K))

    if m.split_space:
        kernel = m.kernel_s
        Kmm_s = gpflow.covariances.Kuu(m.Zs, kernel, jitter=gpflow.default_jitter())
        Kmn_s = gpflow.covariances.Kuf(m.Zs, kernel, m.Xs_mean)    

    # fk(xk)
    for k in range(m.K):
        kernel = m.kernel_K[k]
        Kmm = gpflow.covariances.Kuu(m.Zp, kernel, jitter=gpflow.default_jitter())
        Kmn = gpflow.covariances.Kuf(m.Zp, kernel, m.Xp_mean)
        if m.split_space:
            Kmm += Kmm_s
            Kmn += Kmn_s
        pred = tf.transpose(Kmn) @ tf.linalg.inv(Kmm) @ m.q_mu[k] # [N, D]
        if by_K:
            pred_Y_k[..., k] = pred.numpy()
        assignment = m.pi.numpy()[:, k]
        pred_Y += pred.numpy() * np.stack([assignment for _ in range(m.D)], axis=1)

    if by_K:
        return pred_Y, pred_Y_k
    else:
        return pred_Y


def get_pred_Y_approx(m, by_K=False):
    pred_Y = np.zeros((m.N, m.D))

    if by_K:
        pred_Y_k = np.zeros((m.N, m.D, m.K))

    # fs(xk)
    Kmm_s = gpflow.covariances.Kuu(m.Zs, m.kernel_s, jitter=gpflow.default_jitter())
    Kmn_s = gpflow.covariances.Kuf(m.Zs, m.kernel_s, m.Xs_mean) 
    pred_s = (tf.transpose(Kmn_s) @ tf.linalg.inv(Kmm_s) @ m.q_mu_s).numpy()

    # fk(xk)
    for k in range(m.K):
        kernel = m.kernel_K[k]
        Kmm = gpflow.covariances.Kuu(m.Zp, kernel, jitter=gpflow.default_jitter())
        Kmn = gpflow.covariances.Kuf(m.Zp, kernel, m.Xp_mean)
        pred = tf.transpose(Kmn) @ tf.linalg.inv(Kmm) @ m.q_mu[k] # [N, D]
        if by_K:
            pred_Y_k[..., k] = pred.numpy()
        assignment = m.pi.numpy()[:, k]
        pred_Y += pred.numpy() * np.stack([assignment for _ in range(m.D)], axis=1)
    pred_Y += pred_s

    if by_K:
        return pred_Y, pred_Y_k, pred_s
    else:
        return pred_Y


def klu(m):
    KL_u = 0
    prior_Kuu = np.zeros((m.M, m.M))
    if m.split_space:
        prior_Kuu += gpflow.covariances.Kuu(m.Zs, m.kernel_s, jitter=gpflow.default_jitter())
    for k in range(2):
        prior_Kuu_k = gpflow.covariances.Kuu(m.Zp, m.kernel_K[k], jitter=gpflow.default_jitter())
        KL_u += gpflow.kullback_leiblers.gauss_kl(q_mu=m.q_mu[k], q_sqrt=m.q_sqrt[k], K=prior_Kuu+prior_Kuu_k)
    return KL_u


def klc(m):
    return m.kl_categorical(m.pi, m.pi_prior)


def klxp(m):
    return m.kl_mvn(m.Xp_mean, m.Xp_var, m.Xp_prior_mean, m.Xp_prior_var)


def klxs(m):
    return m.kl_mvn(m.Xs_mean, m.Xs_var, m.Xs_prior_mean, m.Xs_prior_var)


def calc_VE_fs(Y, Fs):
    sample_VE = np.zeros((Y.shape[0]))
    for n in range(Y.shape[0]):
        y = Y[n, :]
        fs = Fs[n, :]
        sample_VE[n] = explained_variance_score(y_true=y, y_pred=fs)
    sample_VE[sample_VE < 0] = 0
    return sample_VE


def calc_VE_fk(Y, Fk):
    N = Y.shape[0]
    K = Fk.shape[-1]
    sample_VE = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            y = Y[n, :]
            fk = Fk[n, :, k]
            sample_VE[n, k] = explained_variance_score(y_true=y, y_pred=fk)
    sample_VE[sample_VE < 0] = 0
    return sample_VE

'''
    MPPCA
'''
def get_X_mppca(Y, W, R, mu):
    N = Y.shape[0]
    D = W.shape[-1] # number of latent dimensions
    K = W.shape[0]
    # [N, D, K]
    X = np.zeros((N, D, K))
    # iterate over each component
    for k in range(K):
        # embedding based on the ith component
        mean = np.concatenate([mu[k, :][None, :] for _ in range(N)], axis=0)
        Xk = np.dot(Y - mean, W[k, ...])
        # each dimension of X is weighted by responsibility for the kth compoenent
        weights = np.concatenate([R[:, k][:, None] for _ in range(D)], axis=1)
        Xk *= weights
        X[..., k] = Xk
    X = np.sum(X, axis=-1)
    return X


def get_pred_mppca(mppcaX, W, R, mu):
    N = mppcaX.shape[0]
    D = W.shape[1] # number of observed dimensions
    K = W.shape[0]
    pred = np.zeros((N, D, K))
    for k in range(K):
        mean = np.concatenate([mu[k, :][None, :] for _ in range(N)], axis=0)
        #predk = np.dot(mppcaX, np.transpose(W[k, ...])) + mean
        predk = np.transpose(W[k, ...] @ np.linalg.inv(np.transpose(W[k, ...]) @ W[k, ...]) @ np.transpose(mppcaX)) + mean
        weights = np.concatenate([R[:, k][:, None] for _ in range(D)], axis=1)
        predk *= weights
        pred[..., k] = predk
    pred = np.sum(pred, axis=-1)
    return pred
