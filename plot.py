import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gpflow
import numpy as np

from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

    
def plot_qmu(m):
    for i in range(2):
        mu = m.q_mu[i].numpy()
        plt.scatter(mu[:, 0], mu[:, 1])


def plot_Fq(Fq, pi, num_data=200):
    x = np.arange(1, num_data+1)
    y = Fq
    plt.scatter(x, y[:, 0], label='k=1')
    plt.scatter(x, y[:, 1], label='k=2')
    Fq_pi = tf.reduce_sum(Fq * pi, axis=1)
    plt.scatter(x, Fq_pi, label='\sum_k pi_nk * Fnk', alpha=0.5)
    plt.vlines(x=int(num_data * 0.5), ymin=np.min(y), ymax=np.max(y), color='r', label='switch branch')
    plt.vlines(x=int(num_data * 0.3), ymin=np.min(y), ymax=np.max(y), color='m', label='branch pt, k=1')
    plt.vlines(x=int(num_data * 0.8), ymin=np.min(y), ymax=np.max(y), color='m', label='branch pt, k=2')
    plt.xlabel('sample index')
    plt.ylabel('Fnk')
    plt.legend()
    

def plot_Fq_list(Fq, num_data=200):
    idx = np.linspace(0, len(Fq)-1, 4, dtype=int)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs = axs.flatten()

    for i in range(idx.size): 
        x = np.arange(1, num_data+1)
        y = Fq[idx[i]]
        axs[i].scatter(x, y[:, 0], label='k=1')
        axs[i].scatter(x, y[:, 1], label='k=2')
        axs[i].vlines(x=int(num_data * 0.5), ymin=y.min(), ymax=y.max(), color='r', label='switch branch')
        axs[i].vlines(x=int(num_data * 0.3), ymin=y.min(), ymax=y.max(), color='g', label='branch pt, k=1')
        axs[i].vlines(x=int(num_data * 0.8), ymin=y.min(), ymax=y.max(), color='g', label='branch pt, k=2')
        axs[i].set_xlabel('sample index')
        axs[i].set_ylabel('Fnk')
        axs[i].set_title('iteration {}'.format(idx[i]))
    plt.legend()
    plt.tight_layout()
    

def plot_assignment(m, Xmean, ax, title):
    assignment = m.pi.numpy()[:, 0]
    sns.scatterplot(x=Xmean[:, 0], y=Xmean[:, 1], hue=assignment, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('true x1')
    ax.set_ylabel('true x2')


def plot_assignment_true_x1(m, x1, ax, title):
    assignment = m.pi.numpy()[:, 0]
    ax.scatter(x1, assignment, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('true x1')
    ax.set_ylabel('assignment probability for GP 0')


def plot_pred_vs_true(Y_true, Y_pred, ax):
    ax.scatter(Y_true[:, 0], Y_true[:, 1], label='true')
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], label='pred', alpha=0.3)
    ax.legend()
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_title('prediction vs. true, first 2D: R^2={:.4f}'.format(r2_score(Y_true, Y_pred)))


def plot_pred_true_1d(Y_true, Y_pred, dim, ax):
    assert dim >= 1, 'the smallest dimension is 1'
    ax.scatter(Y_true[:, dim-1], Y_pred[:, dim-1], alpha=0.3)
    ax.set_xlabel('true')
    ax.set_ylabel('pred')
    r2 = r2_score(Y_true[:, dim-1], Y_pred[:, dim-1])
    ax.set_title('prediction vs. true, dim={}: R^2={:.4f}'.format(dim, r2))
    

def plot_pred_by_K(Y_pred_k, ax):
    colors = ['r', 'g', 'b']
    for k in range(Y_pred_k.shape[-1]):
        pred = Y_pred_k[..., k]
        ax.scatter(pred[:, 0], pred[:, 1], color=colors[k], label='k={}'.format(k), alpha=0.3)
    ax.legend()
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_title('prediction by mixture')


def plot_predS(pred_s, ax):
    ax.scatter(pred_s[:, 0], pred_s[:, 1], color='m', label='fs', alpha=0.3)
    ax.legend()
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_title('prediction by mixture')


def plot_embedding(m, X, labels):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)
    if m.split_space:
        x1 = m.Xs_mean.numpy().flatten()
        x2 = m.Xp_mean.numpy().flatten()
        axs[0].set_xlabel('shared')
        axs[0].set_ylabel('private')
    else:
        xmean = m.Xp_mean.numpy()
        x1 = xmean[:, 0]
        x2 = xmean[:, 1]
        axs[0].set_xlabel('private 1')
        axs[0].set_ylabel('private 2')
    sns.scatterplot(x=x1, y=x2, hue=labels, ax=axs[0], alpha=0.5)
    axs[0].set_title('color by true assignment')
    sns.scatterplot(x=x1, y=x2, hue=m.pi.numpy()[:, 0], ax=axs[1])
    axs[1].set_title('color by learned assignment')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 0], ax=axs[2])
    axs[2].set_title('color by true x1')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 1], ax=axs[3])
    axs[3].set_title('color by true x2')



def plot_Y(Y, X, labels, Z=None, alpha=0.5):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, ax=axs[0], alpha=alpha)
    if Z is not None:
        sns.scatterplot(x=Z[:, 0], y=Z[:, 1], color='m', ax=axs[0])
    sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=labels, ax=axs[1], alpha=alpha)
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


# d is the dimension of y to plot, 1 is the first dimension
# note this plot only makes sense when we constrain only on x1
def plot_fgx(X, xp, fx, gx, d):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    assert d >= 1, 'first dimension of y is 1'
    for i in range(2):
        axs[i].scatter(X[:, i], fx[:, d-1], marker='.', label='f')
        axs[i].scatter(X[:, i], gx[:, d-1], marker='.', label='g')
        axs[i].set_xlabel('x{}'.format(i+1))
        axs[i].set_ylabel('y{}'.format(d))
        axs[i].vlines(x=xp[i], ymin=fx.min(), ymax=fx.max(), color='r', label='xb')
        axs[i].legend()


def plot_pca(Y, X, labels):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    xmean = PCA(n_components=2).fit_transform(Y)
    x1 = xmean[:, 0]
    x2 = xmean[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=labels, ax=axs[0], alpha=0.5)
    axs[0].set_title('color by true assignment')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 0], ax=axs[1])
    axs[1].set_title('color by true x1')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 1], ax=axs[2])
    axs[2].set_title('color by true x2')
    axs[0].set_xlabel('PC 1')
    axs[0].set_ylabel('PC 2')

def plot_gplvm(m, X, labels):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    xmean = m.X_data_mean.numpy()
    x1 = xmean[:, 0]
    x2 = xmean[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=labels, ax=axs[0], alpha=0.5)
    axs[0].set_title('color by true assignment')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 0], ax=axs[1])
    axs[1].set_title('color by true x1')
    sns.scatterplot(x=x1, y=x2, hue=X[:, 1], ax=axs[2])
    axs[2].set_title('color by true x2')
    axs[0].set_xlabel('latent dimension 1')
    axs[0].set_ylabel('latent dimension 2')


def plot_ve_fk(ve_fk, X, labels):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4), sharex=True, sharey=True)
    axs[0].scatter(X[labels==0, 0], ve_fk[labels==0, 0])
    axs[0].set_title('fk, k=0, blue branch')
    axs[1].scatter(X[labels==1, 0], ve_fk[labels==1, 0])
    axs[1].set_title('fk, k=0, orange branch')
    axs[2].scatter(X[labels==0, 0], ve_fk[labels==0, 1])
    axs[2].set_title('fk, k=1, blue branch')
    axs[3].scatter(X[labels==1, 0], ve_fk[labels==1, 1])
    axs[3].set_title('fk, k=1, orange branch')

    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('variance explained')
    for ax in axs:
        ax.grid()


def plot_ve_fs(ve_fs, X, labels):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axs[0].scatter(X[labels==0, 0], ve_fs[labels==0])
    axs[0].set_title('fs, blue branch')
    axs[1].scatter(X[labels==1, 0], ve_fs[labels==1])
    axs[1].set_title('fs, orange branch')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('variance explained')
    for ax in axs:
        ax.grid()


def plot_xs_xp(m, X, labels):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xs = m.Xs_mean.numpy().flatten()
    xp = m.Xp_mean.numpy().flatten()

    axs[0].scatter(X[labels==0, 0], xs[labels==0], label='branch 1', alpha=0.5)
    axs[0].scatter(X[labels==1, 0], xs[labels==1], label='branch 2', alpha=0.5)
    axs[0].set_xlabel('true x1')
    axs[0].set_ylabel('xs')

    axs[1].scatter(X[labels==0, 1], xp[labels==0], label='branch 1', alpha=0.5)
    axs[1].scatter(X[labels==1, 1], xp[labels==1], label='branch 2', alpha=0.5)
    axs[1].set_xlabel('true x2')
    axs[1].set_ylabel('xp')