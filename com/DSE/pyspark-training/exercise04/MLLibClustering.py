from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('k-means-clustering-iris').getOrCreate()

# Exercise 01

# create new dataframe with a csv
df = spark.read.csv("in/iris_bezdekIris.csv", inferSchema=True)\
    .toDF("sep_len", "sep_wid", "pet_len", "pet_wid", "label")
df.printSchema()

# K-Means Clustering
vector_assembler = VectorAssembler(
    inputCols=["sep_len", "sep_wid", "pet_len", "pet_wid"],
    outputCol="features")
df_temp = vector_assembler.transform(df)

# K-Means Clustering
kmeans = KMeans().setK(3)
kmeans = kmeans.setSeed(1)

# fitting to the model
model = kmeans.fit(df_temp)

# Cluster centers
centers = model.clusterCenters()
print(centers)

plt.scatter(centers[0], centers[1])
plt.show()
