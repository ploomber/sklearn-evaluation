from unittest import TestCase

from sklearn_evaluation import table


class TestMissingInput(TestCase):

    def test_feature_importances(self):
        with self.assertRaisesRegex(ValueError, "needed to tabulate"):
            table.feature_importances(None)
