from pydoc import locate
import inspect
from functools import partial

class TrainedClassificationModel(object):
    '''
        This class encapsulates the results of a model that has been trained, 
        you can pass and instance of this class to ReportGenerator for 
        generating HTML reports. It also performs some basic checks based on 
        the properties (e.g. y_true, y_pred and y_score are the same length, 
        target_names and values in y_pred have same number of different values)
    '''
    #All parameters are optional, when creating a report, ReportGenerator
    #class should be able to figure out what it can generate based on the
    #parameters
    def __init__(self, model=None, y_true=None, y_pred=None, y_score=None,
        feature_names=None, target_names=None, model_name=None):
        self._model = model
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_score = y_score
        self._feature_names = feature_names
        self._target_names = target_names
        self._model_name = model_name
        #TODO: perform basic logic checking,
        #raise Exception if necessary

    #Properties should be read-only to ensure instance integrity
    @property
    def model(self):
        return self._model
    @property
    def y_true(self):
        return self._y_true
    @property
    def y_pred(self):
        return self._y_pred
    @property
    def y_score(self):
        return self._y_score
    @property
    def feature_names(self):
        return self._feature_names
    @property
    def target_names(self):
        return self._target_names
    @property
    def model_name(self):
        return self._model_name

    #This method enables access to the plots, metrics and tables modules
    def __getattr__(self, module_name):
        return ModuleProxy(module_name, self)

class ModuleProxy:
    def __init__(self, module_name, trained_model):
        self.module_name = module_name
        self.trained_model = trained_model
    def __getattr__(self, function_name):
        #Get the corresponding function from the package
        fn =  locate('{}.{}'.format(self.module_name, function_name))
        if fn is None:
            raise Exception("Couldn't locate '{}' in '{}' module"
                .format(function_name, self.module_name))
        #Get the function signature
        fn_args = inspect.getargspec(fn).args
        #Get list of properties in trained_model
        properties = [name[1:] for name in self.trained_model.__dict__.keys()]
        #Get the size of the intersection
        size = len(set(fn_args).intersection(set(properties)))
        #Get values for the properties up until index
        values = [getattr(self.trained_model, key) for key in fn_args[:size]]
        if None in values:
            arg_name = fn_args[values.index(None)]
            raise Exception('{} is needed to compute {}'
                .format(arg_name, function_name))
        #Partially apply function
        return partial(fn, *values)

