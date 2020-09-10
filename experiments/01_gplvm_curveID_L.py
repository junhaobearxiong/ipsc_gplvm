import argparse
import time
import sys

import numpy as np
import pandas as pd
import pickle
import gpflow
import tensorflow as tf

from gpflow.ci_utils import ci_niter

parser = argparse.ArgumentParser(description='run GPLVM on 20K cells w/ num_latent dim = 1 or 2, using all or subset (w/ slingshot pseudotime) of cells')
parser.add_argument('curve_id', type=str, help='1, 2, or all')
parser.add_argument('num_latent_dim', type=int, help='number of latent dimensions for GPLVM')
parser.add_argument('num_inducing', type=int, nargs='?', default=50)
parser.add_argument('maxiter', type=int, nargs='?', default=1, help='max number of iterations for the optimizer')
parser.add_argument('log_freq', type=int, nargs='?', default=1, help='log objective funciton value every `log_freq` iterations')
args = parser.parse_args()

start_time = time.time()
np.random.seed(42)  # fix random initialization
sys.path.append('/home/brucejzxiong2009/split-gpm')

print('******************* GPLVM: curve id={}, num_latent_dim={}, num_inducing={}, maxiter={}, log_freq={} *********************'.format(
    args.curve_id, args.num_latent_dim, args.num_inducing, args.maxiter, args.log_freq))


######### Helpers ##################
def init_model(Y, num_latent_dim=args.num_latent_dim, num_inducing=args.num_inducing):
    num_data = Y.shape[0]  # number of data points

    X_mean_init = gpflow.utilities.ops.pca_reduce(Y, num_latent_dim)
    X_var_init = np.ones((num_data, num_latent_dim))
    Z = np.random.permutation(X_mean_init)[:num_inducing] 

    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * num_latent_dim)

    m = gpflow.models.BayesianGPLVM(
        data=Y,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        inducing_variable=Z
    )
    return m


@tf.function
def optimization_step(model: gpflow.models.BayesianGPLVM, opt: tf.optimizers.Adam):
    opt.minimize(model.training_loss, var_list=model.trainable_variables)
    return model.elbo()


def run_optimizer(model: gpflow.models.BayesianGPLVM, maxiter: int, log_freq: int):
    adam_opt = tf.optimizers.Adam(learning_rate=0.01)

    logf = []
    tol = 1e-4
    log_freq = log_freq
    num_iterations = maxiter

    tf.print('initial elbo {:.4f}'.format(model.elbo()))

    for step in range(num_iterations):
        start_time = time.time()
        elbo = optimization_step(model, adam_opt)
        logf.append(elbo)
        if step > 0 and np.abs(elbo - logf[-2]) < tol:
            tf.print('converge at iteration {} elbo {:.4f}'.format(step+1, elbo))
            break
        if (step + 1)  % log_freq == 0:
            tf.print('iteration {} elbo {:.4f}, took {:.4f}s'.format(step+1, elbo, time.time()-start_time))

    return logf


############# Load Data ###############
print('*************** loading data **************************')

data = pd.read_csv('data/scaled_20Kcells.csv', index_col=0)
pseudotime = pd.read_csv('data/pseudotime_20Kcells.csv', index_col=0)

# select cells from a lineage
if args.curve_id == 'all':
    Y = data.values
else:
    cells = pseudotime[~pseudotime['curve{}'.format(args.curve_id)].isna()].index
    Y = data.loc[cells, :].values


############# Main ###################
m = init_model(Y, args.num_latent_dim, args.num_inducing)
print('**************** start training model *********************')

logf = run_optimizer(m, args.maxiter, args.log_freq)

print('**************** finish training model ********************')
        
with open('outputs/gplvm_L{}_curve{}.pkl'.format(args.num_latent_dim, args.curve_id), 'wb') as f:
    pickle.dump([m.X_data_mean.numpy(), m.X_data_var.numpy(), logf], f)

print('**************** took {:.4f}s ***********************'.format(time.time() - start_time))

