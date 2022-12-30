from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..telemetry import SKLearnEvaluationLogger

import numpy as np
import matplotlib.pyplot as plt


@SKLearnEvaluationLogger.log(feature='plot')
def pca(X, Y=None, n_components=2, standard_scalar=False, ax=None):
    """
    Plot principle component analysis curve.

    Parameters
    ----------
    
    X : array-like, shape = [n_samples, n_features]
        Training data, where n_samples is the number of samples and
        n_features is the number of features
    
    Y : array-like or list or None
        set None if Ignored otherwise, pass the targets here
        
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1
    
    standard_scalar : Flag 
        set True if features needs to be scaled
    
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Notes
    -----
    It is assumed that the X parameter are in array format for features


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/pca.py

    """
    if np.isnan(X).sum():
        raise ValueError('X array should not consist nan '
                         'pca')
        
    if ax is None:
        ax = plt.gca()
    
    # Standardizing the features
    if standard_scalar:
        X = StandardScaler().fit_transform(X)
    
    # PCA Projection to 2D
    _pca(X, Y, n_components, ax)

    return ax


def _pca(X, Y=None, n_components=2, ax=None):
    """
    Plot principle component analysis curve.

    Parameters
    ----------
    
    X : array-like, shape = [n_samples, n_features]
        Training data, where n_samples is the number of samples and
        n_features is the number of features
    
    Y : array-like or list or None
        set None if Ignored otherwise, pass the targets here
    
    n_components : int, float or 'mle', default=None
        Number of components to keep.
    
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X)
    
    if ax is None:
        ax = plt.gca()
    
    targets = None
    if Y is None:
        ax.scatter(principalComponents[:,0],principalComponents[:,1], s = 50)
    else:
        Y = Y if isinstance(Y,np.ndarray) else np.array(Y)
        
        if len(Y.shape)<2:
            Y = np.expand_dims(Y,1)
        new_principalComponents = np.hstack((principalComponents,Y))
        
        from random import randint
        targets = np.unique(new_principalComponents[...,-1])
        colors  = ['#%06X' % randint(0, 0xFFFFFF) for i in range(len(targets))]


        for target, color in zip(targets,colors):
            indicesToKeep = new_principalComponents[...,-1] == target
            
            ax.scatter(principalComponents[indicesToKeep,0],principalComponents[indicesToKeep,1]
               , c = color
               , s = 50)
        
    _set_ax_settings(ax,targets)
    
    return ax


def _set_ax_settings(ax,targets=None):
    ax.set_title('Principal Component Plot', fontsize = 10)
#     ax.set_xlim([-2, 2])
#     ax.set_ylim([-0.5, 5])
    ax.set_xlabel('PC1', fontsize = 10)
    ax.set_ylabel('PC2', fontsize = 10)
    if targets is not None:
        ax.legend(targets)
    ax.grid()
