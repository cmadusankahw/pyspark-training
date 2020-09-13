from pyspark.shell import sqlContext, spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

cleanedTaxes = sqlContext.sql("SELECT * FROM cleaned_taxes")
cleanedTaxes.show()

# %sql
# DROP TABLE IF EXISTS cleaned_taxes;

# CREATE TABLE cleaned_taxes AS
# SELECT
#   state,
#   int(zipcode / 10) as zipcode,
#   int(mars1) as single_returns,
#   int(mars2) as joint_returns,
#   int(numdep) as numdep,
#   double(A02650) as total_income_amount,
#   double(A00300) as taxable_interest_amount,
#   double(a01000) as net_capital_gains,
#   double(a00900) as biz_net_income
# FROM taxes2013

markets = spark.read \
    .option("header", "true") \
    .csv("in/market_data.csv")

# Use `sum` to aggregate all the columns in the `cleanedTaxes` dataset -- NOTE: Some data will be nonsense (
# i.e.summing zipcode) but other data could become useful features (i.e. summing AGI in the zipcode). Group the
# `cleanedTaxes` dataframe by zipcode, then `sum` to aggregate across all columns. Save the resulting dataframe in
# `summedTaxes` `show` the `summedTaxes` dataframe
summedTaxes = cleanedTaxes \
    .groupBy("zipcode") \
    .sum()

summedTaxes.show()

# Group the market data into buckets and count the number of farmer's markets in each bucket. Use `selectExpr` to
# transform the market data into labels that identify which zip group they belong to (we used `int(zip/10)` to group
# the tax data) call the new value `zipcode`.  `selectExpr` is short for "Select Expression" and can process similar
# operations to SQL statements. Group by the `zipcode` you just created, then `count` the groups. Use another
# `selectExpr` to transform the data, you only need to keep the `count` and the `zipcode as zip`. Store the results
# in a new dataset called `cleanedMarkets`. `show` `cleanedMarkets`
cleanedMarkets = markets \
    .selectExpr("*", "int(zip / 10) as zipcode") \
    .groupBy("zipcode") \
    .count() \
    .selectExpr("double(count) as count", "zipcode as zip")

cleanedMarkets.show()

# Join the two cleaned datasets into one dataset for analysis.
# Outer join `cleanedMarkets` to `summedTaxes` using `zip` and `zipcode` as the join variable.
# Name the resulting dataset `joined`.
joined = cleanedMarkets \
    .join(summedTaxes, cleanedMarkets["zip"] == summedTaxes["zipcode"], "outer")

print(joined.take(5))

# MLLib doesn't allow null values.  These values came up as `null` in the join because there were no farmer's markets
# in that zip code "basket".  It makes sense to replace the `null` values with zeros. Use the `na` prefix to `fill`
# the empty cells with `0`. Name the resulting dataset `prepped` and `display` it.
prepped = joined.na.fill(0)
print(prepped.take(5))

#  #### Part Two -Use MLLib with Spark ####

# Put all the features into a single vector. Create an array to list the names of all the **non-feature** columns:
# `zip`, `zipcode`, `count`, call it `nonFeatureCols`. Create a list of names called `featureCols` which excludes the
# columns in `nonFeatureCols`. `print` the `featureCols`.
nonFeatureCols = {'zip', 'zipcode', 'count'}
featureCols = [column for column in prepped.columns if column not in nonFeatureCols]
print(featureCols)

# Use the `VectorAssembler` from `pyspark.ml.feature` to add a `features` vector to the `prepped` dataset.
# Call the new dataset `finalPrep`, then `display` only the `zipcode` and `features` from `finalPrep`.
assembler = VectorAssembler(
    inputCols=[column for column in featureCols],
    outputCol='features')

finalPrep = assembler.transform(prepped)
print(finalPrep.select('zipcode', 'features'))

# Display the feature columns graphed out against each other as a scatter plot (hint: exclude `zip`, `zipcode` and
# `features` using `drop`)
print(finalPrep.drop("zip").drop("zipcode").drop("features"))

# Split the `finalPrep` data set into training and testing subsets.  The sets should be randomly selected,
# 70 percent of the samples should go into the `training` set, and 30 percent should go into the `test` set. Cache
# `training` and `test`. Perform an action such as `count` to populate the cache.
(training, test) = finalPrep.randomSplit((0.7, 0.3))

training.cache()
test.cache()

print(training.count())
print(test.count())

# Spark MLLib supports both `regressors` and `classifiers`, in this example you will use linear regression.  Once you
# create the `regressor` you will train it, and it will return a `Model`. The `Model` will be the object you use to
# make predictions. Create an instance of the `LinearRegression` algorithm called `lrModel`: Set the label column to
# "count" Set the features column to "features" Set the "ElasticNetParam" to 0.5 (this controlls the mix of l1 and l2
# regularization--we'll just use an equal amount of each) Print the results of calling `explainParams` on `lrModel`.
# This will show you all the possible parameters, and whether or not you have customized them.
lrModel = LinearRegression() \
    .setLabelCol("count") \
    .setFeaturesCol("features") \
    .setElasticNetParam(0.5)

print("Printing out the model Parameters:")
print("-" * 20)
print(lrModel.explainParams())
print("-" * 20)

# Use the `fit` method on `lrModel` to provide the `training` dataset for fitting.
lrFitted = lrModel.fit(training)

# Make a prediction by using the `transform` method on `lrFitted`, passing it the `test` dataset.
# Store the results in `holdout`.
# `transform` adds a new column called "prediction" to the data we passed into it.
holdout = lrFitted.transform(test)
print(holdout.select("prediction", "count"))

# The `transform` method shows us how many farmer's markets the `lrFitted` method predicts there will be in each zip
# code based on the features we provided.  The raw predictions are not rounded at all. Use a `selectExpr` to relabel
# `prediction` as `raw_prediction`. `round` the `prediction` and call it `prediction` inside the expression Select
# `count` for comparison purposes. Create a column called `equal` that will let us know if the model predicted
# correctly.
holdout = holdout.selectExpr(
  "prediction as raw_prediction",
  "double(round(prediction)) as prediction",
  "count",
  """CASE double(round(prediction)) = count 
                              WHEN true then 1
                              ELSE 0
                              END as equal""")
print(holdout)

# Use another `selectExpr` to `display` the proportion of predictions that were exactly correct.
print(holdout.selectExpr("sum(equal)/sum(1)"))

# Use `RegressionMetrics` to get more insight into the model performance. NOTE: Regression metrics requires input
# formatted as tuples of `double`s where the first item is the `prediction` and the second item is the observation (
# in this case the observation is `count`).  Once you have `map`ped these values from `holdout` you can directly pass
# them to the `RegressionMetrics` constructor.
mapped = holdout.select("prediction", "count").rdd.map(lambda x: (float(x[0]), float(x[1])))
rm = RegressionMetrics(mapped)

print("MSE: ", rm.meanSquaredError)
print("MAE: ", rm.meanAbsoluteError)
print("RMSE Squared: ", rm.rootMeanSquaredError)
print("R Squared: ", rm.r2)
print("Explained Variance: ", rm.explainedVariance)

# Because these results still aren't very good, rather than training a single-model, let's train several using a
# pipeline. Use a `RandomForestRegressor` algorithm.  This algorithm has several `hyperparameters` that we can tune,
# rather than tune them individually, we will use a `ParamGridBuilder` to search the "hyperparameter space" for us.
# This can take some time on small clusters, so be patient. Use the `Pipeline` to feed the algorithm into a
# `CrossValidator` to help prevent "overfitting". Use the `CrossValidator` uses a `RegressionEvaluator` to test the
# model results against a metric (default is RMSE). NOTE: In production, using AWS EC2 compute-optimized instance
# speed this up -- 3 min (c3.4xlarge) vs 10 min (r3.xlarge)
rfModel = RandomForestRegressor() \
    .setLabelCol("count") \
    .setFeaturesCol("features")

paramGrid = ParamGridBuilder() \
    .addGrid(rfModel.maxDepth, [5, 10]) \
    .addGrid(rfModel.numTrees, [20, 60]) \
    .build()

steps = [rfModel]

pipeline = Pipeline().setStages(steps)

cv = CrossValidator() \
    .setEstimator(pipeline) \
    .setEstimatorParamMaps(paramGrid) \
    .setEvaluator(RegressionEvaluator().setLabelCol("count"))

pipelineFitted = cv.fit(training)

# Access the best model on the `pipelineFitted` object by accessing the first stage of the `bestModel` attribute.
print("The Best Parameters:\n--------------------")
print(pipelineFitted.bestModel.stages[0])

# Use the `bestModel` to `transform` the `test` dataset. Use a `selectExpr` to show the raw prediction,
# rounded prediction, count, and whether or not the prediction exactly matched (hint: this is the same `selectExpr`
# you used on the previous model results). tore the results in `holdout2`, then display.
holdout2 = pipelineFitted.bestModel \
    .transform(test) \
    .selectExpr("prediction as raw_prediction",
                "double(round(prediction)) as prediction",
                "count",
                """CASE double(round(prediction)) = count 
  WHEN true then 1
  ELSE 0
END as equal""")

print(holdout2)

# Show the `RegressionMetrics` for the new model results
mapped2 = holdout2.select("prediction", "count").rdd.map(lambda x: (float(x[0]), float(x[1])))
rm2 = RegressionMetrics(mapped2)

print("MSE: ", rm2.meanSquaredError)
print("MAE: ", rm2.meanAbsoluteError)
print("RMSE Squared: ", rm2.rootMeanSquaredError)
print("R Squared: ", rm2.r2)
print("Explained Variance: ", rm2.explainedVariance)

print(holdout2.selectExpr("sum(equal)/sum(1)"))
