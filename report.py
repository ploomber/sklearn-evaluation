from string import Template
import plots as p
from cStringIO import StringIO
import base64


def generate_report(model, y_true, y_pred, path=None, name=None):
    #Read html template
    file = open('report_template.html', 'r')
    t = Template(file.read())
    #Compute values
    cm = p.confusion_matrix_plot(y_true, y_pred, [1,0])
    cm_base64 = figure2base64(cm)

    d = {'model_name': 'RandomForest',
         'date': 'Today',
         'model_properties': 'Properties',
         'feature_list': '1.2.3.4',
         'feature_importance': '1.2.3',
         'confusion_matrix': cm_base64,
         'roc': 'roc',
         'precision_recall': 'pr',
         }
    #Replace values in template
    t = t.substitute(d)
    #If path is provided, save report to disk
    if path is not None:
        name = d['model_name']+'.html' if name==None else name
        report_file = open(name, 'w')
        report_file.write(t)
        report_file.close()
    else:
        return t


def figure2base64(fig):
    io = StringIO()
    fig.savefig(io, format='png')
    fig_base64 = base64.encodestring(io.getvalue())
    return fig_base64
