import np as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# dataset: Iris Flowers Classification Dataset

# step 1: data loading and analysis
# create spark session
spark = SparkSession.builder.appName('decision-tree-classifier').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv("in/iris_bezdekIris.csv", inferSchema=True)\
    .toDF("sep_len", "sep_wid", "pet_len", "pet_wid", "label")
df.printSchema()

# step 2: Data Preprocessing: Transforming columns with VectorAssembler to generate feature and label columns
# as required by the Classification model
vector_assembler = VectorAssembler(
    inputCols=["sep_len", "sep_wid", "pet_len", "pet_wid"],
    outputCol="features")
df_temp = vector_assembler.transform(df)

# drop unnecessary columns
df = df_temp.drop('sep_len', 'sep_wid', 'pet_len', 'pet_wid')
df.show(3)

# Indexing labels
l_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
df = l_indexer.fit(df).transform(df)
df.show(3)

# splitting dataset into Train Test splits
(train, test) = df.randomSplit([0.7, 0.3])


# step 3: get prediction with DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features")
model = dt.fit(train)

predictions = model.transform(test)
predictions.select("prediction", "labelIndex").show(5)

evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex", predictionCol="prediction",
    metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("(Decision Tree Classifier)")
print("accuracy: ", accuracy)


# step 4: get prediction with RandomForestClassifier
rf = RandomForestClassifier(labelCol="labelIndex",
                            featuresCol="features", numTrees=10)

model = rf.fit(train)
rf_predictions = model.transform(test)
rf_predictions.select("prediction", "labelIndex").show(5)

rf_accuracy = evaluator.evaluate(rf_predictions)
print("(Random Forest Classifier)")
print(" accuracy: ", rf_accuracy)

# step 5: get prediction with NaiveBayesClassifier

# Before fitting data into NaiveBayesClassifier, train and test splits should be splited
# in a different way to reach most accurate predictions
nb_splits = df.randomSplit([0.6, 0.4], 1234)
nb_train = nb_splits[0]
nb_test = nb_splits[1]

nb = NaiveBayes(labelCol="labelIndex",
                featuresCol="features", smoothing=1.0,
                modelType="multinomial")

model = nb.fit(nb_train)
nb_predictions = model.transform(nb_test)
nb_predictions.select("label", "labelIndex", "probability", "prediction").show()

nb_accuracy = evaluator.evaluate(nb_predictions)
print("(Naive Bayes Classifier)")
print("accuracy = " + str(nb_accuracy))
