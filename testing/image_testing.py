import os
from matplotlib.testing.compare import compare_images

def equal_images(expected, actual, tol=13, basepath=None):
    '''
        Compare to images
    '''
    #If user provided a basepath, build path from it
    #otherwise, asume it provided full paths to files
    if basepath is not None:
        expected = os.path.join(basepath, expected)
        actual = os.path.join(basepath, actual)

    result = compare_images(expected=expected, actual=actual, tol=tol, in_decorator=True)
    if result is None:
        return True
    else:
        return False
