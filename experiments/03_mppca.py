import pandas as pd
import numpy as np
import argparse
import time
import sys

from utils import *

start_time = time.time()
sys.path.append('/home/brucejzxiong2009/split-gpm')

parser = argparse.ArgumentParser()
parser.add_argument('maxiter', type=int, nargs='?', default=1, help='max number of iterations for the optimizer')
parser.add_argument('subset', type=bool, nargs='?', default=True, help='whether to subset the cells and genes')
parser.add_argument('num_cells', type=int, nargs='?', default=50, help='if subset, how many cells to retain')
parser.add_argument('num_genes', type=int, nargs='?', default=10, help='if subet, how many genes to retain')
args = parser.parse_args()

data = pd.read_csv('data/scaled_20Kcells.csv', index_col=0)
Y = data.values

if args.subset:
	# subset to more variable genes and randomly sample cells
	gene_std = data.std(axis=0)
	gene_subset = gene_std[gene_std > np.quantile(gene_std, 1 - args.num_genes / Y.shape[1])].index
	data_subset = data.loc[:, gene_subset].sample(n=args.num_cells, random_state=1)
	Y = data_subset.values

print(Y.shape)
m1 = init_split_gplvm(
    Y=Y,
    split_space=False,
    Qp=1,
    K=2,
    kernel_K=[gpflow.kernels.Linear() for _ in range(2)],
    M=30
)
set_trainable(m1.Zp, False)

elbo = train_natgrad_adam(m1, False, args.maxiter, 1)

print('**************** took {:.4f}s ***********************'.format(time.time() - start_time))