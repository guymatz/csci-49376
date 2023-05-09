#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import when
import sys

DATA_FILE="data-test.csv"
DATA_FILE="data.csv"

# Create a Spark Session
if 'spark' not in dir():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

# Load and parse the data file into an RDD
data = spark.read.csv(DATA_FILE,header=True, sep=",", inferSchema=True)
print("--- Data with original tex label: \n", data.take(1))

# Change diagnosis to 0/1
data = data.withColumn("diagnosis",
                       when(data.diagnosis == "M", 1)
                      .when(data.diagnosis == "B", 0)
       )
print("--- Data with label replaced with 0/1: \n", data.take(1))
# Split the data into training and test sets (30% held out for testing)
labeled_data = data.rdd.map(lambda row:LabeledPoint(row[1], row[2:]))
print("--- labeled data: \n", labeled_data.take(1))
(training_data, test_data) = labeled_data.randomSplit([0.7, 0.3])
#test_data = labeled_data
#training_data = labeled_data
print("--- test data: \n", test_data.take(1))

# Train a RandomForest model.
model = RandomForest.trainClassifier(training_data,
            numClasses=2, categoricalFeaturesInfo={},
            numTrees=3, featureSubsetStrategy="auto",
            impurity='gini', maxDepth=4, maxBins=6)
print("--- A bit of the model: \n", model.toDebugString().split('\n')[0:10])

# Suggested approach whch does not work
# predictionAndLabels = test_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))
# predictionAndLabels.first()


#sys.exit(1)
# Evaluate model on test instances and compute test error
### Old approach
predictions = model.predict(test_data.map(lambda x: x.features))
print("prediction:\n", predictions.take(3))
#print("prediction:\n", predictions.collect())
test_data_labels = test_data.map(lambda lp: lp.label)
#print(test_data.take(3))
print("test data: \n", test_data.collect())
#print(test_data_labels.take(3))
#print("test labels: \n", test_data_labels.collect())
labels_predictions = test_data_labels.zip(predictions)
#print(labels_predictions.take(3)) 
print("label predictions: \n", labels_predictions.collect()) 
#sys.exit(1)
testErr = labels_predictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
print('Test Error = ' + str(testErr))
#print('Learned classification forest model:')
#print(model.toDebugString())
#
## Save and load model
#model.save(sc, "myRandomForestClassificationModel")
#sameModel = RandomForestModel.load(sc, "myRandomForestClassificationModel")
