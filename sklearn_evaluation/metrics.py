import numpy as np
from sklearn.metrics import precision_score

def precision_at(labels, scores, percent=0.01):
    '''
    Calculates precision at a given percent.
    Only supports binary classification.
    '''
    #Sort scores in descending order    
    scores_sorted = np.sort(scores)[::-1]

    #Based on the percent, get the index to split the data
    #if value is negative, return 0
    cutoff_index = max(int(len(labels) * percent) - 1, 0)
    #Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]

    #Convert scores to binary, by comparing them with the cutoff value
    scores_binary = map(lambda x: int(x>=cutoff_value), scores)
    #Calculate precision using sklearn function
    precision = precision_score(labels, scores_binary)

    return precision, cutoff_value