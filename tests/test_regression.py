
import numpy as np
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot

y_true = np.array([105, 120, 120, 160, 120, 145, 175, 160, 185, 210, 150])
y_pred = np.array([108.19, 115.16, 122.13, 136.06, 136.06, 156.97, 163.94, 170.91, 184.84, 205.75, 151.23])

@image_comparison(baseline_images=['residual'],
                  extensions=['png'],
                  remove_text=True)
def test_residuals():
    plot.residuals(y_true, y_pred)

@image_comparison(baseline_images=['prediction_error'],
                  extensions=['png'],
                  remove_text=True)
def test_prediction_error():
    plot.prediction_error(y_true, y_pred)
