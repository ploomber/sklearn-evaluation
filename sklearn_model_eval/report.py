from string import Template
import plots as p
from cStringIO import StringIO
import base64
import re
import os
import mistune
from datetime import datetime

class ReportGenerator:
    def __init__(self, savepath, template, css):
        pass

    def generate_report(model, y_true, y_pred, y_score, feature_list, target_names, name):
        pass


def generate_report(model, y_true, y_pred, y_score, feature_list, target_names, path=None, name=None):
    #Read md template and compile to html
    pkg = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(pkg, 'templates', 'classification_default.md')
    f = open(filepath, 'r')
    md = f.read()
    markdown = mistune.Markdown()
    html = markdown(md)

    #Add css
    filepath = os.path.join(pkg, 'templates', 'github.css')
    f = open(filepath, 'r')
    css = f.read()
    html = '<style>'+css+'</style>'+html
    t = Template(html)

    #Feature importance
    try:
        fi,table = p.feature_importance_plot(model, feature_list)
        fi_base64 = figure2base64(fi)
        fi_warning = ''
    except:
        fi = ''
        fi_base64 = ''
        fi_warning = 'This model does not have feature importances.'

    #Confusion matrix
    cm = p.confusion_matrix_(y_true, y_pred, target_names)
    cm_base64 = figure2base64(cm)
    #ROC
    roc = p.roc(y_true, y_score)
    roc_base64 = figure2base64(roc)
    #Precision-Recall
    pr = p.precision_recall(y_true, y_score)
    pr_base64 = figure2base64(pr)

    d = {'model_name': get_model_name(model),
         'date': datetime.now().strftime('%B %d %Y %H:%M'),
         'model_properties': prettify_dict(model.get_params()),
         'feature_list':  prettify_list(feature_list),
         'feature_importance_plot':  base64_2_html(fi_base64),
         'feature_importance_warning': fi_warning,
         'confusion_matrix': base64_2_html(cm_base64),
         'roc': base64_2_html(roc_base64),
         'precision_recall': base64_2_html(pr_base64),
         }

    #Replace values in template
    t = t.substitute(d)
    #If path is provided, save report to disk
    if path is not None:
        name = d['model_name']+'.html' if name==None else name
        report_file = open(path+name, 'w')
        report_file.write(t)
        report_file.close()
    else:
        return t

def base64_2_html(img):
    return '<img src="data:image/png;base64,'+img+'"></img>'

def figure2base64(fig):
    io = StringIO()
    fig.savefig(io, format='png')
    fig_base64 = base64.encodestring(io.getvalue())
    return fig_base64

def prettify_list(l):
    l = [str(idx+1)+'. '+str(el) for idx,el in enumerate(l)]
    return reduce(lambda x,y:x+'<br>'+y, l)

def prettify_dict(d):
    return prettify_list([key+': '+str(d[key]) for key in d.keys()])

def get_model_name(model):
    s = str(type(model))
    model_name = re.search(".*'(.+?)'.*", s).group(1).split(".")[-1]
    return model_name
