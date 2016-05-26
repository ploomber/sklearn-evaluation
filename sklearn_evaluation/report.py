from string import Template
import plots
from cStringIO import StringIO
import base64
import os
from datetime import datetime
from utils import get_model_name

try:
    import mistune
except:
    raise Exception('You need to install mistune to use the report module')


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class ReportGenerator:
    '''
        Generate HTML reports from ClassificationModelResults instances.
        If savepath is None, generate_report() will return HTML code,
        otherwise, HTML file will be saved there.
    '''
    def __init__(self, savepath=None):
        self.savepath = savepath
    def __call__(self, trained_model):
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
        model_class_name = get_model_name(trained_model.model)
        #If trained_model does not have a model_name, use model_class_name
        model_name = model_class_name if trained_model.model_name is None else trained_model.model_name
    
        #Feature importance
        try:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            plots.feature_importance(trained_model.model,
                                     trained_model.feature_names,
                                     ax=ax)
            fi_content = figure2html(fig)
        except AttributeError:
            fi_content = '%s does not support feature importances' % (model_class_name)
        except TypeError:
            fi_content = 'To compute this plot you need to provide a model and a feature list'
        except:
            fi_content = 'An unkwown error happened while computing feature importances plot'

        #Confusion matrix
        try:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            plots.confusion_matrix(trained_model.y_true,
                                   trained_model.y_pred,
                                   ax,
                                   trained_model.target_names)
            cm_content = figure2html(fig)
        except TypeError:
            cm_content = 'To compute this plot you need to provide y_true, y_pred and target_names'
        except:
            cm_content = 'An unkwown error happened while computing the confusion matrix'

        #ROC
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.roc(trained_model.y_true, trained_model.y_score, ax=ax)
        roc_base64 = figure2base64(fig)
        #Precision-Recall
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.precision_recall(trained_model.y_true, trained_model.y_score, ax=ax)
        pr_base64 = figure2base64(fig)
    
        d = {'model_name': model_name,
             'date': datetime.now().strftime('%B %d %Y %H:%M'),
             'model_properties': prettify_dict(trained_model.model.get_params()),
             'feature_names':  prettify_list(trained_model.feature_names),
             'feature_importance_plot':  fi_content,
             'confusion_matrix': cm_content,
             'roc': base64_2_html(roc_base64),
             'precision_recall': base64_2_html(pr_base64),
             }
    
        #Replace values in template
        t = t.substitute(d)
        #If path is provided, save report to disk
        if self.savepath is not None:
            path_to_report = os.path.join(self.savepath, model_name+'.html')
            report_file = open(path_to_report, 'w')
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