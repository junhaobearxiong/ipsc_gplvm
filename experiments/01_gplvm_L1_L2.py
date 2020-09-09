import argparse
import time
import sys

import numpy as np
import pandas as pd
import pickle
import gpflow
from gpflow.ci_utils import ci_niter

parser = argparse.ArgumentParser(description='run GPLVM on 20K cells w/ latent dim = 1 or 2, using all or subset (w/ slingshot pseudotime) of cells')
parser.add_argument('maxiter', type=int, nargs='?', default=1)
parser.add_argument('num_inducing', type=int, nargs='?', default=50)
args = parser.parse_args()

start_time = time.time()
np.random.seed(42)  # fix random initialization
sys.path.append('/home/brucejzxiong2009/split-gpm')

print('******************* GPLVM w/ {} inducing variables, training maxiter={} *********************'.format(args.num_inducing, args.maxiter))

######### Helpers ##################
def init_model(Y, latent_dim, num_inducing=args.num_inducing):
    num_data = Y.shape[0]  # number of data points

    X_mean_init = gpflow.utilities.ops.pca_reduce(Y, latent_dim)
    X_var_init = np.ones((num_data, latent_dim))
    Z = np.random.permutation(X_mean_init)[:num_inducing] 

    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * latent_dim)

    m = gpflow.models.BayesianGPLVM(
        data=Y,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        inducing_variable=Z
    )
    return m


############# Load Data ###############
print('*************** loading data **************************')

data = pd.read_csv('data/scaled_20Kcells.csv', index_col=0)
pseudotime = pd.read_csv('data/pseudotime_20Kcells.csv', index_col=0)


############# Main ###################

# select cells from a lineage
for curve_id in ['1', '2', 'all']:
    if curve_id == 'all':
        Y = data.values
    else:
        cells = pseudotime[~pseudotime['curve{}'.format(curve_id)].isna()].index
        Y = data.loc[cells, :].values
    
    for latent_dim in [1, 2]:
        print('**************** start training model: curve id={}, L={} *********************'.format(curve_id, latent_dim))
        m = init_model(Y, latent_dim, args.num_inducing)
        opt = gpflow.optimizers.Scipy()
        _ = opt.minimize(
            m.training_loss,
            method="BFGS",
            variables=m.trainable_variables,
            options=dict(maxiter=ci_niter(args.maxiter)),
        )
        print('**************** finish training model: curve id={}, L={} ********************'.format(curve_id, latent_dim))
        print('**************** took {}s'.format(time.time() - start_time))
        with open('outputs/gplvm_L{}_curve{}.pkl'.format(latent_dim, curve_id), 'wb') as f:
            pickle.dump([m.X_data_mean.numpy(), m.X_data_var.numpy()], f)

print('*********************** finish! total time {}s'.format(time.time() - start_time))