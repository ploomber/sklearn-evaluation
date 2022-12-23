
# %% tags=["parameters"]
model = None
params = None
upstream = None
product = None

# %%
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv(upstream['data']['data'])

# %%
X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=0)

# %% tags=["model"]
mod, _, attr = model.rpartition('.')
reg = getattr(importlib.import_module(mod), attr)(**params)
reg.fit(X_train, y_train)
print(model)

# %% tags=["params"]
print(reg.get_params())

# %% tags=["mse"]
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)
