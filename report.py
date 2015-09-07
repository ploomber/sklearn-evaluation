from string import Template
import plots as p
from cStringIO import StringIO
import base64
import re
from datetime import datetime

def generate_report(model, y_true, y_pred, y_score, feature_list, target_names, path=None, name=None):
    #Read html template
    file = open('report_template.html', 'r')
    t = Template(file.read())
    #Confusion matrix
    cm = p.confusion_matrix_plot(y_true, y_pred, target_names)
    cm_base64 = figure2base64(cm)
    #ROC
    roc = p.roc_plot(y_true, y_score)
    roc_base64 = figure2base64(roc)
    #Precision-Recall
    pr = p.precision_recall_plot(y_true, y_score)
    pr_base64 = figure2base64(pr)

    d = {'model_name': get_model_name(model),
         'date': datetime.now().strftime('%B %d %Y %H:%M'),
         'model_properties': prettify_dict(model.get_params()),
         'feature_list':  prettify_list(feature_list),
         'feature_importance_plot': '1.2.3',
         'confusion_matrix': cm_base64,
         'roc': roc_base64,
         'precision_recall': pr_base64,
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

def prettify_list(l):
    l = [str(idx+1)+'. '+el for idx,el in enumerate(l)]
    return reduce(lambda x,y:x+'<br>'+y, l)

def prettify_dict(d):
    return prettify_list([key+': '+str(d[key]) for key in d.keys()])

def get_model_name(model):
    s = str(type(model))
    model_name = re.search(".*'(.+?)'.*", s).group(1).split(".")[-1]
    return model_name
