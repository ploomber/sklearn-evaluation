from string import Template
import plots as p
from cStringIO import StringIO
import base64
import os
import mistune
from datetime import datetime
from utils import get_model_name

class ReportGenerator:
    def __init__(self, savepath=None, template=None, css=None):
        self.savepath = savepath
    def generate_report(self, model, y_true, y_pred, y_score, feature_list, target_names, name):
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

        #Get the placeholders in the template
        #so things that the user do not want are not computed

        #Get name for the model
        model_name = get_model_name(model)
    
        #Feature importance
        try:
            fi = p.feature_importance(model, feature_list)
            fi_content = figure2html(fi)
        except AttributeError:
            fi_content = '%s does not support feature importances' % (model_name)
        except TypeError:
            fi_content = 'To compute this plot you need to provide a model and a feature list'
        except:
            fi_content = 'An unkwown error happened while computing feature importances plot'

        #Confusion matrix
        try:
            cm = p.confusion_matrix_(y_true, y_pred, target_names)
            cm_content = figure2html(cm)
        except TypeError:
            cm_content = 'To compute this plot you need to provide y_true, y_pred and target_names'
        except:
            cm_content = 'An unkwown error happened while computing the confusion matrix'

        #ROC
        roc = p.roc(y_true, y_score)
        roc_base64 = figure2base64(roc)
        #Precision-Recall
        pr = p.precision_recall(y_true, y_score)
        pr_base64 = figure2base64(pr)
    
        d = {'model_name': model_name,
             'date': datetime.now().strftime('%B %d %Y %H:%M'),
             'model_properties': prettify_dict(model.get_params()),
             'feature_list':  prettify_list(feature_list),
             'feature_importance_plot':  fi_content,
             'confusion_matrix': cm_content,
             'roc': base64_2_html(roc_base64),
             'precision_recall': base64_2_html(pr_base64),
             }
    
        #Replace values in template
        t = t.substitute(d)
        #If path is provided, save report to disk
        if self.savepath is not None:
            name = d['model_name']+'.html' if name==None else name
            report_file = open(self.savepath+name, 'w')
            report_file.write(t)
            report_file.close()
        else:
            return t

def figure2html(fig):
    return base64_2_html(figure2base64(fig))

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