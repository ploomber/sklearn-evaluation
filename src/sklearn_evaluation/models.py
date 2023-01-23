from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, auc
import numpy as np
from sklearn_evaluation import plot
import re
from sklearn_evaluation.report.serialize import EvaluatorHTMLSerializer
from sklearn_evaluation.report.report import Report
from jinja2 import Template
from sklearn_evaluation.report.serialize import figure2html
from sklearn_evaluation.evaluator import _gen_ax


class Range(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def in_range(self, n) -> bool:
        return self.min <= n and self.max >= n


def _check_array_balance(array) -> bool:
    classes, counts = np.unique(array, return_counts=True)
    n_classes = len(classes)
    n_values = len(array)
    expected_balance = 1 / n_classes

    weights = list(map(lambda count: count / n_values, counts))
    balance_threshold = 0.05
    expected_range = Range(
        expected_balance - balance_threshold, expected_balance + balance_threshold
    )

    return all(expected_range.in_range(w) for w in weights)


def _create_report_template(evaluation_state):
    title = "Model evaluation"
    print(evaluation_state)

    template = Template(
        """
<html>
  <head>
      <style>
        .model-evaluation-container {
            font-family: Arial, Helvetica, sans-serif;
            text-align: left;
            width: fit-content;
            margin: 50px auto;
        }

        .block {
            margin-bottom: 50px
        }

        .nobull {
            list-style-type: none;
        }

        ul li {
            margin-bottom: 10px;
        }

        </style>

    </head>
  <body>
    <div class="model-evaluation-container">
        <div>
            <h1>{{title}}</h1>

            <div class="block">

                {% if not evaluation_state["balance"]["is_ok"] %}
                <ul>
                    <li class="nobull"><h2>Balance</h2></li>
                    {% for guideline in evaluation_state["balance"]["guidelines"] %}

                        {% if guideline is string %}
                            <li>{{guideline}}</li>
                        {% else %}
                            <p>{{figure2html(guideline.get_figure())}}</p>
                        {% endif %}
                    {% endfor %}
                </ul>

                {% endif %}

            </div>

            <div class="block">

                {% if not evaluation_state["accuracy"]["is_ok"] %}
                <ul>
                    <li class="nobull"><h2>Accuracy</h2></li>
                    {% for guideline in evaluation_state["accuracy"]["guidelines"] %}

                        {% if guideline is string %}
                            <li>{{guideline}}</li>
                        {% else %}
                            <p>{{figure2html(guideline.get_figure())}}</p>
                        {% endif %}
                    {% endfor %}
                </ul>

                {% endif %}

            </div>

            <div class="block">

                {% if not evaluation_state["auc"]["is_ok"] %}
                <ul>
                    <li class="nobull"><h2>AUC</h2></li>
                    {% for guideline in evaluation_state["auc"]["guidelines"] %}

                        {% if guideline is string %}
                            <li>{{guideline}}</li>
                        {% else %}
                            <p>{{figure2html(guideline.get_figure())}}</p>
                        {% endif %}
                    {% endfor %}
                </ul>

                {% endif %}
            </div>


            <div class="block">

                <ul>
                <li class="nobull"><h2>General Stats</h2></li>
                {% for guideline in evaluation_state["general_stats"]["guidelines"] %}

                    {% if guideline is string %}
                        <li>{{guideline}}</li>
                    {% else %}
                        <p>{{figure2html(guideline.get_figure())}}</p>
                    {% endif %}
                {% endfor %}
                </ul>

            </div>


        </div>
    </div>
  </body>
</html>
    """
    )
    return template.render(
        title=title, evaluation_state=evaluation_state, figure2html=figure2html
    )


class ModelEvaluator(object):
    @staticmethod
    def evaluate_balance(y_true, evaluation_state):
        """
        Return guidelines
        """
        balance_state = dict()
        guidelines = []

        # y_true
        is_balanced = _check_array_balance(y_true)

        if is_balanced:
            balance_state["is_ok"] = True
            guidelines.append("your model is balanced")
        else:
            balance_state["is_ok"] = False
            p = plot.target_analysis(y_true)
            guidelines.append("Your test set is highly imbalanced")
            guidelines.append(p)
            guidelines.append(
                "To tackle this, check out this "
                "<a href='https://ploomber.io/blog/' target='_blank'>guide</a>"
            )

        balance_state["guidelines"] = guidelines
        evaluation_state["balance"] = balance_state

    @staticmethod
    def evaluate_accuracy(y_test, y_pred_test, evaluation_state):
        """
        This score measures how many labels the
        model got right out of the total number of predictions
        """
        accuracy_threshold = 0.9
        accuracy_state = dict()

        guidelines = []
        accuracy = accuracy_score(y_test, y_pred_test)

        balance = evaluation_state["balance"]

        guidelines.append(f"Accuracy is {accuracy}")
        if accuracy >= accuracy_threshold:
            if balance["is_ok"]:
                accuracy_state["is_ok"] = True
                guidelines.append("You model is accurate")
            else:
                accuracy_state["is_ok"] = False
                guidelines.append(
                    "Please note your model is unbalanced, "
                    "so high accuracy could be misleading"
                )

        accuracy_state["guidelines"] = guidelines
        evaluation_state["accuracy"] = accuracy_state

    @staticmethod
    def evaluate_confusion_matrix(y_true, y_pred, evaluation_state):
        """
        How many of a classifier's predictions were correct,
        and when incorrect, where the classifier got confused
        """
        cm = confusion_matrix(y_true, y_pred)

        # normalize
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # go check values
        # diagonal = np.diagonal(cm)

    @staticmethod
    def evaluate_auc(y_true, y_score, evaluation_state):
        """ """
        auc_state = dict()
        auc_state["is_ok"] = False
        guidelines = []
        auc_threshold_low_range = Range(0, 0.6)
        auc_threshold_acceptable_range = Range(0.7, 0.8)

        # auc - roc
        roc = plot.ROC.from_raw_data(y_true, y_score)
        for i in range(len(roc.fpr)):
            roc_auc = auc(roc.fpr[i], roc.tpr[i])
            print()
            # todo: better check
            label = roc.label[i] if len(roc.label) > 0 else f"class {i}"
            r = re.match(r"^\(class (.)*\)", label)
            if r:
                class_name = r[0].replace("(", "").replace(")", "")
            else:
                class_name = label  # f"class {i}"

            if auc_threshold_low_range.in_range(roc_auc):
                guidelines.append(f"Your area under {class_name} curve is low")
                class_roc = plot.ROC(roc.fpr[i], roc.tpr[i], label=[label]).plot().ax
                guidelines.append(class_roc)

                guidelines.append(
                    "To tackle this, check out this "
                    "<a href='https://ploomber.io/blog/' target='_blank'>guide</a>"
                )
            elif auc_threshold_acceptable_range.in_range(roc_auc):
                auc_state["is_ok"] = True
                guidelines.append(f"AUC FOR CLASS {class_name} IS ACCEPTABLE")
            else:
                auc_state["is_ok"] = True
                guidelines.append(f"AUC FOR CLASS {class_name} IS EXECLLENT")

        auc_state["guidelines"] = guidelines
        evaluation_state["auc"] = auc_state

    @staticmethod
    def generate_general_stats(y_true, y_pred, y_score, evaluation_state):
        general_stats = dict({"guidelines": []})

        general_stats["guidelines"].append(
            plot.confusion_matrix(y_true, y_pred, ax=_gen_ax())
        )
        general_stats["guidelines"].append(plot.roc(y_true, y_score, ax=_gen_ax()))
        evaluation_state["general_stats"] = general_stats


def evaluate_model(
    y_true,
    y_pred,
    model,
    y_score=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):

    # validate if inputs are ok
    _check_model(model)
    # TODO: if optional args given test them.
    # skip for now
    _check_inputs(y_true, y_pred)

    # start model evaluation
    evaluation_state = dict()

    # check imbalance
    ModelEvaluator.evaluate_balance(y_true, evaluation_state)

    # accuracy score
    ModelEvaluator.evaluate_accuracy(y_true, y_pred, evaluation_state)

    # confusion matrix
    # ModelEvaluator.evaluate_confusion_matrix(y_true, y_pred, evaluation_state)

    # auc
    ModelEvaluator.evaluate_auc(y_true, y_score, evaluation_state)

    # add general stats
    ModelEvaluator.generate_general_stats(y_true, y_pred, y_score, evaluation_state)

    e = EvaluatorHTMLSerializer(None)
    template = _create_report_template(evaluation_state)
    report = Report(e, template)
    return report


def _check_model(model) -> None:
    """
    Validate if model supported

    Raises
    ~~~~~~
    ModelNotSupported or ValueError?
    """


def _check_inputs(y_true, y_pred) -> None:
    """
    Validate if inputs supported

    Raises
    ~~~~~~
    ModelNotSupported or ValueError?
    """
