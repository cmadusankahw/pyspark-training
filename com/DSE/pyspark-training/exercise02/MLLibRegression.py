from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator



# dataset: the top 5 Fortune 500 companies in the year 2017

# create spark session
spark = SparkSession.builder.appName('ml-fortune-500').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv('in/power_plant.csv', header=True, inferSchema=True).cache()
df.printSchema()

vectorAssembler = VectorAssembler(inputCols=["AT", "V", "AP", "RH"],
                                  outputCol="features")

vpp_df = vectorAssembler.transform(df)
print(vpp_df.take(1))

# step 1: building and training Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="PE")

lr_model = lr.fit(vpp_df)

# view coefficients
print(lr_model.coefficients)

# view intercept
print(lr_model.intercept)  # X coordinate where the line crosses the Y axis

# model summary
print(lr_model.summary)

# errors
print(lr_model.summary.rootMeanSquaredError)

##############################################################

# step 2: Decision Tree regression model
# splitting data into train/ test sets to predict
splits = vpp_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

print(train_df.count(), test_df.count())

# building and training  model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
dt_model = dt.fit(train_df)

# get predictions
dt_predictions = dt_model.transform(test_df)

# evaluating results
dt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")  # evaluating root mean squared error
rmse = dt_evaluator.evaluate(dt_predictions)
print(rmse)



