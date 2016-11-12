from . import compute


__all__ = ['feature_importances']


# http://ipython.readthedocs.org/en/stable/config/integrating.html
class Table():
    def __init__(self, content, header):
        try:
            self._tabulate = __import__('tabulate').tabulate
        except:
            raise ImportError('tabulate is required to use the table module')
        self.content = content
        self.header = header

    @property
    def html(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='html')

    def __str__(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='grid')

    def _repr_html_(self):
        return self.html


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
    table
        Table object with the data. Columns are
        feature_name, importance (`std_` only included for models with
        sub-estimators)

    """
    res = compute.feature_importances(data, top_n, feature_names)
    return Table(res, res.dtype.names)
