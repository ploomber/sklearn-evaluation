import numpy as np
from sklearn.metrics import precision_score

def precision_at(labels, scores, percent=0.01):
    '''
    Calculates precision at a given percent. Only supports binary classification.
    '''
    cutoff_index = max(int(len(labels) * percent) - 1, 0)
    
    scores_sorted = np.sort(scores)[::-1]
    cutoff_value = scores_sorted[cutoff_index]

    scores_binary = map(lambda x: int(x>=cutoff_value), scores)

    precision = precision_score(labels, scores_binary)
    return precision, cutoff_value