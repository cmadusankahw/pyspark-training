from pyspark.sql import SparkSession
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
import pandas as pd

# dataset: the top 5 Fortune 500 companies in the year 2017

# create spark session
spark = SparkSession.builder.appName('ml-fortune-500').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv('in/fortune500.csv', header=True, inferSchema=True).cache()
df.printSchema()

# Step 1: Data exploration

# To find out if any of the variables, fields have correlations or dependencies, we can plot a scatter matrix
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']

sampled_data = df.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n - 1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

plt.show()

# Step 2: Data Pre-processing

# Using VectorAssembler, Assemble columns in to feature and label columns to train the model
vectorAssembler = VectorAssembler(inputCols=['Rank', 'Employees'], outputCol='features')
tcompany_df = vectorAssembler.transform(df)
tcompany_df = tcompany_df.select(['features', 'Employees'])
tcompany_df.show(3)

# Step 3: Building the Linear Model

# splitting data to test , train data sets
splits = tcompany_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# another approach to divide test, train splits
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.3)

# building Linear Regression model
# maxIter = No of maximum iterations
lr = LinearRegression(featuresCol='features', labelCol='Employees', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# plotting coefficients
beta = np.sort(lr_model.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# make predictions
predictions = lr_model.transform(test_df)
predictions.show(10)

# Model summary
summary = lr_model.binarySummary
print(summary)

# evaluating predictions
evaluator = BinaryClassificationEvaluator()
print('Evaluations: ', evaluator.evaluate(predictions))

