import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn_evaluation import plot
from sklearn.cluster import KMeans

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

categories = [
    "comp.graphics", "comp.windows.x", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware"
]

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
t0 = time()
X_tfidf = vectorizer.fit_transform(dataset.data)
lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
X_lsa = lsa.fit_transform(X_tfidf)

kmeans = KMeans(
    n_clusters=true_k,
    max_iter=100,
    n_init=5,
)

plot.silhouette_analysis(X_lsa,
                         kmeans,
                         range_n_clusters=[2, 3, 4, 5, 6],
                         metric='cosine')
plt.show()
