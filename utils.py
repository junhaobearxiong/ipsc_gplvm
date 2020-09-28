import numpy as np
import gpflow
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, List
import time

from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.models import gplvm
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable

from split_gplvm import SplitGPLVM
from test_gplvm import TestGPLVM


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


# generate high dimensional observation Y ([num_data, dim])
# by mapping X to dim-dimensional space through a GP w/ kernel function
def gen_obs(X, split_space, labels, dim, lengthscales, obs_noise=1):
    N = X.shape[0]
    Y = np.zeros([N, dim])
    
    k1 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales[0])
    k2 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales[1])
    
    label_counts = np.unique(labels, return_counts=True)[1]
    
    # y = fs(x1) + fk(x2), where k depends on the label of x
    # each fk has a different lengthscale
    if split_space:
        k3 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales[2])
        # fs(x1)
        Y += np.random.default_rng().multivariate_normal(np.zeros(N), k3(X[:, 0][:, np.newaxis]), dim).T
        # fk(x2), k=1
        Y[labels == 0, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[0]), 
                                                                       k1(X[labels == 0, 1][:, np.newaxis]), dim).T
        # fk(x2), k=2
        Y[labels == 1, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[1]),
                                                                       k2(X[labels == 1, 1][:, np.newaxis]), dim).T
    else:
        # fk(x), k=1
        Y[labels == 0, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[0]), k1(X[labels == 0, :]), dim).T
        # fk(x), k=2
        Y[labels == 1, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[1]), k2(X[labels == 1, :]), dim).T
    # add observational noise
    Y += np.random.normal(0, np.sqrt(obs_noise), Y.shape)
    return Y #(Y - Y.mean()) / Y.std()


def gen_Y(X, labels, dim, kernels, obs_noise=0):
    N = X.shape[0]
    Y = np.zeros([N, dim])
    
    label_counts = np.unique(labels, return_counts=True)[1]
    # fk(x), k=1
    Y[labels == 0, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[0]), 
                                                                     kernels[0](X[labels == 0, :]), dim).T
    # fk(x), k=2
    Y[labels == 1, :] += np.random.default_rng().multivariate_normal(np.zeros(label_counts[1]), 
                                                                     kernels[1](X[labels == 1, :]), dim).T
    # add observational noise
    Y += np.random.normal(0, np.sqrt(obs_noise), Y.shape)
    return Y


def gen_Y_single_kernel(X, dim, kernel, obs_noise):
    Y = np.random.default_rng().multivariate_normal(np.zeros(X.shape[0]), kernel(X), dim).T
    Y += np.random.normal(0, np.sqrt(obs_noise), Y.shape)
    return Y


def plot_Y(Y, X, labels, Z=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, ax=axs[0])
    if Z is not None:
        sns.scatterplot(x=Z[:, 0], y=Z[:, 1], color='m', ax=axs[0])
    sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=labels, ax=axs[1])
    sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=X[:, 0], ax=axs[2])

    axs[0].set_title('Latent space')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')

    axs[1].set_title('Observed space: 2D, color by labels')
    axs[1].set_xlabel('y1')
    axs[1].set_ylabel('y2')

    axs[2].set_title('Observed space: 2D, color by x1')
    axs[2].set_xlabel('y1')
    axs[2].set_ylabel('y2')


def plot_kernel_samples(k, ax, xmin=-3, xmax=3):
    xx = np.linspace(xmin, xmax, 100)[:, None]
    K = k(xx)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)


def init_gplvm(Y, latent_dim, kernel, num_inducing=None, inducing_variable=None, X_mean_init=None, X_var_init=None):
    num_data = Y.shape[0]  # number of data points

    if X_mean_init is None:
        X_mean_init = tf.constant(PCA(n_components=latent_dim).fit_transform(Y), dtype=default_float())
    if X_var_init is None:
        X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())

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
    if X_var_init is None:
        X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())

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


def init_split_gplvm(Y, split_space, Qp, K, kernel_K=None, M=None, Xp_mean_init=None, Xp_var_init=None,
    Zp=None, Zs=None, pi_init=None, Qs=None):
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
        pi_init = tf.ones((N, K), dtype=default_float())

    if split_space:
        Xs_mean_init = tf.constant(X_pca[:, :Qs], dtype=default_float())
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
        )
    else:
        model = SplitGPLVM(
            data=Y,
            split_space=False,
            Xp_mean=Xp_mean_init,
            Xp_var=Xp_var_init,
            pi=pi_init,
            kernel_K=kernel_K,
            Zp=Zp
        )
    
    return model


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
        


def train_natgrad_adam(model, num_iterations=2000, log_freq=10):
    
    natgrad_opt = NaturalGradient(gamma=1.0)
    adam_opt = tf.optimizers.Adam(learning_rate=0.01)
    variational_params = list(zip(model.q_mu, model.q_sqrt))
    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)

    @tf.function
    def optimization_step():
        natgrad_opt.minimize(model.training_loss, var_list=variational_params)
        adam_opt.minimize(model.training_loss, var_list=model.trainable_variables)
        return (model.elbo(), model.Fq)
        #return model.elbo()

    log_elbo = []
    log_Fq = []
    # log_predY = []
    tol = 1e-4

    print('initial elbo {:.4f}'.format(model.elbo()))

    for step in range(num_iterations):
        start_time = time.time()
        elbo, Fq = optimization_step()
        #elbo = optimization_step()
        log_elbo.append(elbo)
        log_Fq.append(Fq.numpy())
        # log_predY.append(pred_Y.numpy())

        if step > 0 and np.abs(elbo - log_elbo[-2]) < tol:
            print('converge at iteration {} elbo {:.4f}'.format(step+1, elbo))
            break
        if (step + 1)  % log_freq == 0:
            print('iteration {} elbo {:.4f}, took {:.4f}s'.format(step+1, elbo, time.time()-start_time))
            
    return (log_elbo, log_Fq)
    #return log_elbo