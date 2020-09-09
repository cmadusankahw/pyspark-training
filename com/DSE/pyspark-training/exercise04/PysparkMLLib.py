from pyspark.sql import SparkSession
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd

# Used Dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution.
# The classification goal is to predict whether the client will subscribe (Yes/No) to a term deposit.
# Dataset Url: https://www.kaggle.com/rouseguy/bankbalanced/data

# create spark session
spark = SparkSession.builder.appName('ml-bank').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv('in/bank.csv', header=True, inferSchema=True)

# Step 1: Analysis of Dataset

# Analysing numeric features and generating basic statistical summary
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']

summary = df.select(numeric_features).describe().toPandas().transpose()
print(summary)

# plot summary data to visualize correlations between column pairs
numeric_data = df.select(numeric_features).toPandas()
axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8))
n = len(numeric_data.columns)  # no of columns

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

plt.show()  # show scatter plot

# there aren’t highly correlated numeric variables. all of them are kept in the model
#  day and month columns are not really useful so they can be removed

# Step 2: Data Preprocessing

# selecting only relevant columns to build the machine learning model
df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')
cols = df.columns

df.printSchema()





