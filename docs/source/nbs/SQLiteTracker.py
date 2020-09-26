# # Tracking Machine Learning experiments

from sklearn_evaluation.manage.SQLiteTracker import SQLiteTracker

# + tags=["parameters"]
tracker = SQLiteTracker(':memory:')

# +
tracker

# +
uuid = tracker.new()

# +
tracker.update(uuid, {'accuracy': 0.85})

# +
tracker

# +
tracker.comment(uuid, 'My experiment')

# +
tracker
