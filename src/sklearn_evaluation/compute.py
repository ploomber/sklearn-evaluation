import warnings
import numpy as np


def feature_importances(data, top_n=None, feature_names=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array-like structure.

    If data is a scikit-learn model with sub-estimators (e.g. RandomForest,
    AdaBoost) the function will compute the standard deviation of each
    feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature_names

    Returns
    -------
    numpy structured array
        Returns a numpy structured array with the data. Columns are
        feature_name, importance and std_ if an sklearn model with
        sub-estimators was passed in data.

    """
    # data can be either a sklearn estimator or an iterator with
    # the actual importances values, try to get the values
    try:
        imp = data.feature_importances_
    except AttributeError:
        imp = np.array(data)

    # in case the user passed an estimator, it may have an estimators_
    # attribute, which includes importances for every sub-estimator
    # get them if possible
    try:
        sub_imp = np.array([e.feature_importances_ for e in data.estimators_])
        # calculate std
        std = np.std(sub_imp, axis=0)
    except Exception:
        std = None

    # get the number of features
    n_features = len(imp)

    # check that the data has the correct format
    if top_n and top_n > n_features:
        top_n = n_features
        warnings.warn(
            ('top_n ({}) is greater than the number of'
             ' features ({}), showing all features'.format(top_n, n_features)))
    if top_n and top_n < 1:
        raise ValueError('top_n cannot be less than 1')
    if feature_names and len(feature_names) != n_features:
        raise ValueError(('feature_names ({}) must match the number of'
                          ' features ({})'.format(len(feature_names),
                                                  n_features)))

    # if the user did not pass feature names create generic names
    if feature_names is None:
        feature_names = [
            'Feature {}'.format(n) for n in range(1, n_features + 1)
        ]
        feature_names = np.array(feature_names)
    else:
        feature_names = np.array(feature_names)

    # order the data according to the importance for the feature
    idx = np.argsort(imp)[::-1]
    imp = imp[idx]
    feature_names = feature_names[idx]
    if std is not None:
        std = std[idx]

    # build the structured array
    if std is not None:
        names = 'feature_name,importance,std_'
        res = np.core.records.fromarrays([feature_names, imp, std],
                                         names=names)
    else:
        names = 'feature_name,importance'
        res = np.core.records.fromarrays([feature_names, imp], names=names)

    # get subset if top_n is not none
    if top_n:
        res = res[:top_n]

    return res
