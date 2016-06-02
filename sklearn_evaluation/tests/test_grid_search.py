import json
import unittest

from matplotlib.testing.decorators import image_comparison, cleanup

from sklearn_evaluation import plot
from sklearn_evaluation.util import _grid_scores_from_dicts

# Parameters:
# n_estimators': [1, 10, 50, 100]
# criterion': ['gini', 'entropy']
# max_features': ['sqrt', 'log2']

with open('static/sample_scores.json') as f:
    grid_scores = _grid_scores_from_dicts(json.loads(f.read()))

with open('static/sample_scores_2_params.json') as f:
    grid_scores_2_params = _grid_scores_from_dicts(json.loads(f.read()))

# plot tests


@image_comparison(baseline_images=['single_numeric_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_line():
    to_vary = 'n_estimators'
    plot.grid_search(grid_scores, to_vary, kind='line')


@image_comparison(baseline_images=['single_numeric_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_line_with_tuple():
    to_vary = ('n_estimators')
    plot.grid_search(grid_scores, to_vary, kind='line')


@image_comparison(baseline_images=['single_numeric_bar'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_bar():
    to_vary = 'n_estimators'
    plot.grid_search(grid_scores, to_vary, kind='bar')


@image_comparison(baseline_images=['single_categorical_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_categorial_line():
    to_vary = 'n_estimators'
    plot.grid_search(grid_scores, to_vary, kind='line')


@image_comparison(baseline_images=['single_categorical_bar'],
                  extensions=['png'],
                  remove_text=True)
def test_single_categorial_bar():
    to_vary = 'n_estimators'
    plot.grid_search(grid_scores, to_vary, kind='bar')


@image_comparison(baseline_images=['single_numeric_restricted_single'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_restricted_single():
    to_vary = 'n_estimators'
    to_keep = {'max_features': 'sqrt', 'criterion': 'gini'}
    plot.grid_search(grid_scores, to_vary, to_keep, kind='bar')


@image_comparison(baseline_images=['single_numeric_restricted_multi'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_restricted_multi():
    to_vary = 'n_estimators'
    to_keep = {'max_features': ['sqrt', 'log2'], 'criterion': 'gini'}
    plot.grid_search(grid_scores, to_vary, to_keep, kind='bar')


@image_comparison(baseline_images=['double'],
                  extensions=['png'],
                  remove_text=True)
def test_double_ignores_kind_line():
    to_vary = ('n_estimators', 'criterion')
    to_keep = {'max_features': 'sqrt'}
    plot.grid_search(grid_scores, to_vary, to_keep, kind='line')


@image_comparison(baseline_images=['double'],
                  extensions=['png'],
                  remove_text=True)
def test_double_ignores_kind_bar():
    to_vary = ('n_estimators', 'criterion')
    to_keep = {'max_features': 'sqrt'}
    plot.grid_search(grid_scores, to_vary, to_keep, kind='bar')


# API tests

class TestGridSearchAPI(unittest.TestCase):

    @cleanup
    def test_list_with_len_three_raises_exception(self):
        l = ['a', 'b', 'c']
        with self.assertRaises(ValueError):
            plot.grid_search(grid_scores, l)

    @cleanup
    def test_none_to_vary_raises_exception(self):
        with self.assertRaises(ValueError):
            plot.grid_search(grid_scores, None)

    @cleanup
    def test_can_send_tuple_len_one(self):
        to_vary = ('n_estimators')
        plot.grid_search(grid_scores, to_vary)

    @cleanup
    def test_can_send_string(self):
        to_vary = 'n_estimators'
        plot.grid_search(grid_scores, to_vary)

    @cleanup
    def test_raise_exception_when_parameter_set_is_not_fully_specified(self):
        with self.assertRaises(ValueError):
            to_vary = ('n_estimators', 'criterion')
            plot.grid_search(grid_scores, to_vary=to_vary, to_keep=None)

    @cleanup
    def test_to_keep_can_be_none_when_parameter_set_is_fully_specified(self):
        to_vary = ('n_estimators', 'criterion')
        plot.grid_search(grid_scores_2_params, to_vary=to_vary, to_keep=None)

    @cleanup
    def test_raise_exception_when_parameter_does_not_exist(self):
        with self.assertRaises(ValueError):
            to_vary = ('this_is_not_a_parameter')
            to_keep = {'criterion': 'gini',
                       'max_features': 'sqrt'}
            plot.grid_search(grid_scores, to_vary=to_vary, to_keep=to_keep)

    @cleanup
    def test_raise_exception_when_parameter_does_not_exist_double(self):
        with self.assertRaises(ValueError):
            to_vary = ('n_estimators', 'this_is_not_a_parameter')
            to_keep = {'criterion': 'gini',
                       'max_features': 'sqrt'}
            plot.grid_search(grid_scores, to_vary=to_vary, to_keep=to_keep)

    @cleanup
    def test_raise_exception_when_invalid_value_in_to_keep_double(self):
        with self.assertRaises(KeyError):
            to_vary = ('n_estimators', 'max_features')
            to_keep = {'criterion': 'not_a_value'}
            plot.grid_search(grid_scores, to_vary=to_vary, to_keep=to_keep)

    @cleanup
    def test_raise_exception_when_invalid_value_in_to_keep(self):
        with self.assertRaises(KeyError):
            to_vary = 'n_estimators'
            to_keep = {'criterion': 'not_a_value'}
            plot.grid_search(grid_scores, to_vary=to_vary, to_keep=to_keep)
