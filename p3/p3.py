#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import when

DATA_FILE="data.csv"

# Create a Spark Session
ss = SparkSession.builder.master("local[*]").getOrCreate()

# Load and parse the data file into an RDD
data = spark.read.csv(DATA_FILE,header=True, sep=",")


#  APPROACH 1
# I don't think we need IDs
data = data.drop(data.id)
# Change diagnosis to 0/1
data = data.withColumn("diagnosis",
                       when(data.diagnosis == "M", 1)
                      .when(data.diagnosis == "B", 0)
       )
# Split the data into training and test sets (30% held out for testing)
labeled_data = data.rdd.map(lambda row:LabeledPoint(row[1], [row[2:]]))
(training_data, test_data) = labeled_data.randomSplit([0.7, 0.3])

### APPROACH 2
##from pyspark.ml.feature import VectorAssembler
### Then something like
##va = VectorAssembler(inputCols = data.columns, outputCol='features')
### see https://www.datatechnotes.com/2021/12/mllib-random-forest-classification.html


# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(training_data,
            numClasses=2, categoricalFeaturesInfo={},
            numTrees=3, featureSubsetStrategy="auto",
            impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(test_data.map(lambda x: x.features))
labels_predictions = test_data.map(lambda lp: lp.label).zip(predictions)
testErr = labels_predictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

# Save and load model
model.save(sc, "myRandomForestClassificationModel")
sameModel = RandomForestModel.load(sc, "myRandomForestClassificationModel")
