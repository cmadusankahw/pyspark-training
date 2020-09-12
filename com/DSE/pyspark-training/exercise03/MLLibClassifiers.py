import np as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# dataset: Iris Flowers Classification Dataset

# step 1: data loading and analysis
# create spark session
spark = SparkSession.builder.appName('decision-tree-classifier').getOrCreate()

# create new dataframe with a csv
df = spark.read.csv("in/iris_bezdekIris.csv", inferSchema=True)\
    .toDF("sep_len", "sep_wid", "pet_len", "pet_wid", "label")
df.printSchema()

###########################################################

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

#####################################################

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

##########################################################

# step 4: get prediction with RandomForestClassifier
rf = RandomForestClassifier(labelCol="labelIndex",
                            featuresCol="features", numTrees=10)

model = rf.fit(train)
rf_predictions = model.transform(test)
rf_predictions.select("prediction", "labelIndex").show(5)

rf_accuracy = evaluator.evaluate(rf_predictions)
print("(Random Forest Classifier)")
print(" accuracy: ", rf_accuracy)

####################################################

# step 5: get prediction with NaiveBayesClassifier

# Before fitting data into NaiveBayesClassifier, train and test splits should be splited
# in a different way to reach most accurate predictions
nb_splits = df.randomSplit([0.6, 0.4], 1)
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

####################################################

# step 6: Getting Predictions with Multilayer Perceptron Classifier

# Defining layers for Multilayer Perceptron classifier
# 4 as we hve 4 features in dataset --> input layer | and 3 for there are 3 categories --> output layer
# (2 layers with 5 neurons in each in the middle)
layers = [4, 5, 5, 3]

# building and fitting the model
mlp = MultilayerPerceptronClassifier(layers= layers, seed= 1)
mlp_model = mlp.fit(nb_train)

# getting predictions
mlp_predictions = mlp_model.transform(nb_test)

# evaluating the model
mlp_accuracy = evaluator.evaluate(mlp_predictions)
print("(Multilayer Perceptron Classifier)")
print("accuracy = " + str(mlp_accuracy))




