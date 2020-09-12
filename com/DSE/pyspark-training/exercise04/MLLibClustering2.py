import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt


# Exercise 02:

spark = SparkSession.builder.appName('k-means-clustering-nd').getOrCreate()

n_samples = 10000
n_features = 3
X, y = make_blobs(n_samples=n_samples, centers=10, n_features=n_features, random_state=42)

# add a row index as a string
pddf = pd.DataFrame(X, columns=['x', 'y', 'z'])
pddf['id'] = 'row'+pddf.index.astype(str)

# move it first (left)
cols = list(pddf)
cols.insert(0, cols.pop(cols.index('id')))
print(pddf.head())

# save the ndarray as a csv file
pddf.to_csv('in/input.csv', index=False)

# plot in a 3-dimensional space
threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
threedee.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.show()

