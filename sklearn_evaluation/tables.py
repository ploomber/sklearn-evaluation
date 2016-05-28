import numpy as np

try:
    from tabulate import tabulate
except:
    Exception('tabulate is required to use the tables module')


# http://ipython.readthedocs.org/en/stable/config/integrating.html
class Table():
    def __init__(self, content, header):
        self.content = content
        self.header = header

    def __str__(self):
        return tabulate(self.content, headers=self.header, tablefmt='grid')

    def _repr_html_(self):
        return tabulate(self.content, headers=self.header, tablefmt='html')


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def _compute_feature_importances(model, feature_names):
    # Get feature importances
    importances = model.feature_importances_
    # Compute standard deviation
    std = np.std([est.feature_importances_ for est in model.estimators_],
                 axis=0)
    # Get indexes in order
    indices = np.argsort(importances)[::-1]
    # Generate lists
    f = [{'name': feature_names[i], 'importance':importances[i], 'std': std[i]} for i in indices]
    return f


def feature_importances(model, feature_names=None, n=None):
    # If no feature_names is provided, assign numbers
    total_features = len(model.feature_importances_)
    feature_names = range(total_features) if feature_names is None else feature_names
    # Plot all features if n is not provided, otherwise plot top n features
    n = len(feature_names) if n is None else n
    # Return only top n features
    data = _compute_feature_importances(model, feature_names)[:n]
    # Convert list of dictionaries to list of tuples
    # that's to show the columns in the correct order: name, importance, std
    rows = [(dic['name'], dic['importance'], dic['std']) for dic in data]
    header = ['name', 'importance', 'std']
    # Remove num key, that's not useful
    return Table(rows, header)
