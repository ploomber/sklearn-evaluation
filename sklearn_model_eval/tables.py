import numpy as np
from tabulate import tabulate

#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances(model, feature_list, format='grid'):
    #Get feature importances
    importances = model.feature_importances_
    #Compute standard deviation
    std = np.std([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
    #Get indexes in order
    indices = np.argsort(importances)[::-1]
    #Generate lists
    f = [{'num':i, 'name':feature_list[i], 'importance':importances[i], 'std': std[i]} for i in indices]
    if format is not None:
        f = tabulate(f, tablefmt=format)
    return f
