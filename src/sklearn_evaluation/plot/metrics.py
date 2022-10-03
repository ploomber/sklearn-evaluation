import numpy as np
from sklearn_evaluation.metrics import compute_at_thresholds
from sklearn_evaluation.plot.util import set_default_ax
from ..telemetry import SKLearnEvaluationLogger


@set_default_ax
@SKLearnEvaluationLogger.log(feature='plot')
def metrics_at_thresholds(fn, y_true, y_score, n_thresholds=10, start=0.0,
                          ax=None):
    """Plot metrics at increasing thresholds
    """
    th, m = compute_at_thresholds(fn, y_true, y_score, n_thresholds,
                                  start)

    ax.plot(th, np.array(m).T, '.--')
    ax.legend([fn_.__name__ for fn_ in fn])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric value')
    ax.grid()

    return ax
