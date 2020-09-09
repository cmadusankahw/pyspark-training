from pyspark.sql import SparkSession
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd

# create spark session
spark = SparkSession.builder.appName('ml-bank').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv('in/RealEstate.csv', header=True, inferSchema=True)

# print df schema
df.printSchema()

# create a Pandas DataFrame with pySpark df
# Transpose method will transform columns as rows if called
pdDf = pd.DataFrame(df.take(5), columns=df.columns).transpose()

# printing dataframe
# Another approach: pdDf.head()
print(pdDf)

# get summary of all numeric (int) features
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
summaryDf = df.select(numeric_features).describe().toPandas().transpose()

print(summaryDf)

# correlation between data
numeric_data = df.select(numeric_features).toPandas()

# plotting with pandas.plotting.scatter_matrix
axs = scatter_matrix(numeric_data, figsize=(8, 8))

n = len(numeric_data.columns)

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# showing the correlation scatter plot
plt.show()

