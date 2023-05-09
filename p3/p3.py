#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import when
import sys

DATA_FILE="data-test.csv"
DATA_FILE="data.csv"
DATA_FILE="labeled_data.svm"

# Create a Spark Session
if 'spark' not in dir():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

# Load and parse the data file into an RDD
labeled_data = MLUtils.loadLibSVMFile(spark.sparkContext,
                                      path='labeled_data.svm')
# Split the data into training and test sets (30% held out for testing)
print("--- labeled data point: \n", labeled_data.take(1))
(training_data, test_data) = labeled_data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
model = RandomForest.trainClassifier(training_data,
            numClasses=2, categoricalFeaturesInfo={},
            numTrees=3, featureSubsetStrategy="auto",
            impurity='gini', maxDepth=4, maxBins=6)

# Evaluate model on test instances and compute test error
predictions = model.predict(test_data.map(lambda x: x.features))
print("prediction:\n", predictions.take(1))
test_data_labels = test_data.map(lambda lp: lp.label)
print("test label: \n", test_data_labels.first())
labels_predictions = test_data_labels.zip(predictions)
#print(labels_predictions.take(3)) 
print("label prediction: \n", labels_predictions.first()) 
testErr = labels_predictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
print('Test Error = ' + str(testErr))
#print('Learned classification forest model:')
#print(model.toDebugString())
#
## Save and load model
#model.save(sc, "myRandomForestClassificationModel")
#sameModel = RandomForestModel.load(sc, "myRandomForestClassificationModel")
