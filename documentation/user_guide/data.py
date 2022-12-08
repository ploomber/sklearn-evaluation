
# %% tags=["parameters"]
upstream = None
product = None

# %%
from sklearn import datasets

# %%
ca_housing = datasets.fetch_california_housing(as_frame=True)
df = ca_housing['frame']
df.to_csv(product['data'], index=False)
