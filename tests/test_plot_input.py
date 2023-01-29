from unittest import TestCase

from sklearn_evaluation import plot

# Testing missing input - checking that meaningful errors are raised when
# trying to plot somethign with missing input (specially relevant when)
# using the report generator


class TestMissingInput(TestCase):
    def test_confusion_matrix(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.confusion_matrix(None, [1, 0])

    def test_roc(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.roc([1, 0], None)

    def test_precision_recall(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.precision_recall(None, None)

    def test_precision_recall_from_raw_data(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.PrecisionRecall.from_raw_data(None, None)

    def test_precision_recall_constructor(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.PrecisionRecall(None, None, None).plot()

    def test_feature_importances(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.feature_importances(None)

    def test_precision_at_proportions(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.precision_at_proportions(None, [1, 0])

    def test_calibration_curve(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.calibration_curve(None, None)

    def test_calibration_curve_from_raw_data(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.CalibrationCurve.from_raw_data(None, None)

    def test_calibration_curve_constructor(self):
        with self.assertRaisesRegex(ValueError, "needed to plot"):
            plot.CalibrationCurve(None, None).plot()
