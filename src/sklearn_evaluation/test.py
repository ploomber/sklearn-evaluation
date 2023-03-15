'''
from sklearn_evaluation.tracker import SQLiteTracker
from tracker import SQLiteTracker

tracker = SQLiteTracker('some.db')
exp = tracker.new_experiment()
#tracker.update(exp.uuid, dict(a=1,b=2,c=3,d=4))
tracker.upsert_append(exp.uuid, dict(a=1, b=3, e='mc^2'))
data = tracker.get(exp.uuid)
print(data)
'''
#import tracker
from tracker import SQLiteTracker
#from sklearn_evaluation import SQLiteTracker
tracker = SQLiteTracker('experiments.db')
exp1 = tracker.new_experiment()
exp1.log("accuracy", 0.8) # doctest: +SKIP
tracker.upsert_append(exp1.uuid, dict(loss=0.2))