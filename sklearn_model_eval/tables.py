import numpy as np

#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances(model, feature_list):
    #Get feature importances
    importances = model.feature_importances_
    #Compute stantard deviation
    std = np.std([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
    #Get indexes in order
    indices = np.argsort(importances)[::-1]
    #Generate lists
    f = [{'num':i, 'name':feature_list[i], 'importance':importances[i], 'std': std[i]} for i in indices]
    return f
