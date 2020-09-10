from pyspark.sql import SparkSession
from pyspark import SparkContext
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd

# create spark session
sc = SparkContext("local", "rdd-training")

################################################################

# Exercise 1: create RDD with parallelizing a list
words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"]
).cache()

print(words.take(3))
print(words.count())


# Print element with foreach
def f(x): print(x)


# in the foreach loop a function is called on every element in rdd
words.foreach(lambda x: f(x))
# or more simpler
words.foreach(f)

# applying filter transformation
words_filter = words.filter(lambda y: 'spark' in y)
filtered = words_filter.collect()
print("Filtered: " + filtered)

# applying map transformation
words_map = words.map(lambda x: (x, 1))
mapped = words_map.collect()
print(" Mapped Key-Value pairs: " + mapped)

##################################################################

# Exercise 2 : Create RDD from a file input
# Read raw text to RDD
lines = sc.textFile('in/airports.text')

# formatting & splitting values in each row
airportRdd = lines.map(lambda x: (x.replace('"', '').split(",")))

print(type(airportRdd))
print(airportRdd.take(2))

##################################################################

# Exercise 3: WordCount in pyspark with reduce & GroupByKey
# Read RDD from a text file
wc_lines = sc.textFile('in/word_count.text')

# flatten the String regular RDD into a list RDD of words
words = wc_lines.flatMap(lambda line: line.split(" "))

# count words to a map with countByValue
wordCounts = words.countByValue()

# count words with reduce() and groupByKey() transformations

wordCounts.foreach(f)

####################################################################

# Exercise 4: join() operation
x = sc.parallelize([("spark", 1), ("hadoop", 4)])
y = sc.parallelize([("spark", 2), ("hadoop", 5)])
joined = x.join(y)
final_joined = joined.collect()
print(final_joined)

##################################################################