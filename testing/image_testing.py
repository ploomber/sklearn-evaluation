from matplotlib.testing.compare import compare_images

def equal_images(expected, actual, tol=13):
    '''
        Compare to images
    '''
    result = compare_images(expected=expected, actual=actual, tol=50, in_decorator=True)
    if result is None:
        return True
    else:
        return False
