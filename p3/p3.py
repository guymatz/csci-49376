#!/usr/bin/env python
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.classification import LogisticRegressionWithLBFGS


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils
# Needed for converting csv to libsvm (no longer needed)
#from pyspark.mllib.regression import LabeledPoint
import sys

DATA_FILE="data-test.csv"
DATA_FILE="data.csv"
DATA_FILE="labeled_data.svm"

# Create a Spark Session
if 'sc' not in dir():
    sc = SparkContext("local","proj_3")

# Load and parse the data file into an RDD
labeled_data = MLUtils.loadLibSVMFile(sc, path='labeled_data.svm')
# Split the data into training and test sets (30% held out for testing)
#print("--- labeled data point: \n", labeled_data.take(1))
(training_data, validation_data, test_data) = labeled_data.randomSplit([0.6, 0.2, 0.2])

test_data_labels = test_data.map(lambda lp: lp.label)
validation_data_labels = validation_data.map(lambda lp: lp.label)
#print("test label: \n", test_data_labels.first())
#print("validation label: \n", validation_data_labels.first())

# Grid is a dict of dicts
# Returns the best-performing hyper-parameters
def find_best_params(grid):
  max_f1 = 0
  best_params = None
  for param_1 in grid.keys():
    for param_2 in grid[param_1]:
      #print(f"Comparing {param_2} {grid[param_1][param_2]} {max_f1}")
      if grid[param_1][param_2] > max_f1:
        max_f1 = grid[param_1][param_2]
        best_params = (param_1, param_2)
  return best_params 

### RANDOM FOREST
# Train a RandomForest model.
rf_grid = {}
for n_trees in range(40, 61, 10):
  rf_grid[n_trees] = {}
  print(f"Running RF with num trees: {n_trees}")
  for max_depth in range(3,6):
    print(f"Running RF with max_depth: {max_depth}")
    rf_model = RandomForest.trainClassifier(training_data,
                numTrees=n_trees,
                maxDepth=max_depth,
                categoricalFeaturesInfo={},
                numClasses=2
                )

    # Evaluate model on test instances and compute test error
    rf_predictions = rf_model.predict(validation_data.map(lambda x: x.features))
    #print("prediction:\n", predictions.take(1))
    rf_labels_predictions = validation_data_labels.zip(rf_predictions)
    #print(labels_predictions.take(3)) 
    #print("label prediction: \n", labels_predictions.first()) 
    #testErr = labels_predictions.filter(
    #    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
    #print('Test Error = ' + str(testErr))

    #rf_metrics = BinaryClassificationMetrics(rf_labels_predictions)

    ### CONFUSION MATRIX
    # True positives, where label is 1 and prediction is 1
    tp_num = rf_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
    # True negatives, where label is 0 and prediction is 0
    tn_num = rf_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
    # FFalse positives, where label is 0 and prediction is 1
    fp_num = rf_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
    # False negatives, where  and label is 1 prediction is 0
    fn_num = rf_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

    precision = tp_num / (tp_num + fp_num)
    recall = tp_num / (tp_num + fn_num)

    f1 = 2 * (precision * recall) / (precision + recall)

    # true-positive rate
    # DO I NEED sensitivity?
    tpr =  tp_num / (tp_num + fn_num)
    # false-positive rate
    fpr = fp_num / (fp_num + tn_num)

    rf_grid[n_trees][max_depth] = f1

# Best hyper-parameters
print(f"RANDOM FOREST:")
best_hp = find_best_params(rf_grid)
print(f"Best hyper-parameters for RF(n_trees, max_depth) are {best_hp}")
rf_model = RandomForest.trainClassifier(training_data,
             numTrees=best_hp[0],
             maxBins=best_hp[1],
             categoricalFeaturesInfo={},
             numClasses=2
           )
# Evaluate model on test instances and compute test error
rf_predictions = rf_model.predict(test_data.map(lambda x: x.features))
#print("GB prediction:\n", rf_predictions.take(1))
rf_labels_predictions = test_data_labels.zip(rf_predictions)
#print(labels_predictions.take(3)) 
#print("label prediction: \n", rf_labels_predictions.first()) 

rf_metrics = BinaryClassificationMetrics(rf_labels_predictions)

### CONFUSION MATRIX
# True positives, where label is 1 and prediction is 1
tp_num = rf_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
# True negatives, where label is 0 and prediction is 0
tn_num = rf_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
# FFalse positives, where label is 0 and prediction is 1
fp_num = rf_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
# False negatives, where  and label is 1 prediction is 0
fn_num = rf_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

precision = tp_num / (tp_num + fp_num)
recall = tp_num / (tp_num + fn_num)

f1 = 2 * (precision * recall) / (precision + recall)

# true-positive rate
# DO I NEED sensitivity?
tpr =  tp_num / (tp_num + fn_num)
# false-positive rate
fpr = fp_num / (fp_num + tn_num)

print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"F1 score: {f1}")
print(f"Area Under ROC: {rf_metrics.areaUnderROC}")
print()

# See
# https://stackoverflow.com/questions/52847408/pyspark-extract-roc-curve
# https://shihaojran.com/distributed-machine-learning-using-pyspark/
# https://www.kaggle.com/code/palmer0/binary-classification-with-pyspark-and-mllib


### Gradient Boosted

# Train a GB model.
gb_grid = {}
for learning_rate in [0.09, 0.1, 0.11]:
  # print(f"Running GB with Learning Rate: {learning_rate}")
  gb_grid[learning_rate] = {}
  for max_depth in range(2,5):
    # print(f"Running GB with Max Depth: {max_depth}")
    gb_model = GradientBoostedTrees.trainClassifier(training_data,
                categoricalFeaturesInfo={},
                learningRate=learning_rate,
                maxDepth=max_depth)

    # Evaluate model on validation instances and compute validation error
    gb_predictions = gb_model.predict(validation_data.map(lambda x: x.features))
    #print("GB prediction:\n", gb_predictions.take(1))
    gb_labels_predictions = validation_data_labels.zip(gb_predictions)
    #print(labels_predictions.take(3)) 
    #print("label prediction: \n", gb_labels_predictions.first()) 
    #testErr = gb_labels_predictions.filter(
    #    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
    #print('Test Error = ' + str(testErr))

    #metrics = BinaryClassificationMetrics(gb_labels_predictions)

    ### CONFUSION MATRIX
    # True positives, where label is 1 and prediction is 1
    tp_num = gb_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
    # True negatives, where label is 0 and prediction is 0
    tn_num = gb_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
    # FFalse positives, where label is 0 and prediction is 1
    fp_num = gb_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
    # False negatives, where  and label is 1 prediction is 0
    fn_num = gb_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

    precision = tp_num / (tp_num + fp_num)
    recall = tp_num / (tp_num + fn_num)

    f1 = 2 * (precision * recall) / (precision + recall)

    # true-positive rate
    # DO I NEED sensitivity?
    tpr =  tp_num / (tp_num + fn_num)
    # false-positive rate
    fpr = fp_num / (fp_num + tn_num)

#    print(f"GRADIENT BOOSTED:")
#    print(f"precision: {precision}")
#    print(f"recall: {recall}")
#    print(f"F1 score: {f1}")
#    print(f"Area Under ROC: {metrics.areaUnderROC}")

    gb_grid[learning_rate][max_depth] = f1
    # See
    # https://stackoverflow.com/questions/52847408/pyspark-extract-roc-curve
    # https://shihaojran.com/distributed-machine-learning-using-pyspark/
    # https://www.kaggle.com/code/palmer0/binary-classification-with-pyspark-and-mllib

# Best hyper-parameters
best_hp = find_best_params(gb_grid)
print(f"Best hyper-parameters for GB(learningRate, maxDepth) are {best_hp}")
gb_model = GradientBoostedTrees.trainClassifier(training_data,
                categoricalFeaturesInfo={},
                learningRate=best_hp[0],
                maxDepth=best_hp[1])

# Evaluate model on test instances and compute test error
gb_predictions = gb_model.predict(test_data.map(lambda x: x.features))
#print("GB prediction:\n", gb_predictions.take(1))
gb_labels_predictions = test_data_labels.zip(gb_predictions)
#print(labels_predictions.take(3)) 
#print("label prediction: \n", gb_labels_predictions.first()) 

gb_metrics = BinaryClassificationMetrics(gb_labels_predictions)

### CONFUSION MATRIX
# True positives, where label is 1 and prediction is 1
tp_num = gb_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
# True negatives, where label is 0 and prediction is 0
tn_num = gb_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
# FFalse positives, where label is 0 and prediction is 1
fp_num = gb_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
# False negatives, where  and label is 1 prediction is 0
fn_num = gb_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

precision = tp_num / (tp_num + fp_num)
recall = tp_num / (tp_num + fn_num)

f1 = 2 * (precision * recall) / (precision + recall)

# true-positive rate
# DO I NEED sensitivity?
tpr =  tp_num / (tp_num + fn_num)
# false-positive rate
fpr = fp_num / (fp_num + tn_num)

print(f"GRADIENT BOOSTED:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
print(f"Area Under ROC: {gb_metrics.areaUnderROC}")
print()

### Logistic Regression with LBGFS
lr_grid = {}
for reg_param in [0.09,0.1,0.11]:
  print(f"Running LR with regParam: {reg_param}")
  lr_grid[reg_param] = {}
  for iterations in range(50,151,50):
    print(f"Running LR with iterations: {iterations}")
    lr_model = LogisticRegressionWithLBFGS.train(training_data,
                regParam=reg_param, iterations=iterations)

    # Evaluate model on test instances and compute test error
    lr_predictions_int = lr_model.predict(validation_data.map(lambda x: x.features))
    lr_predictions = lr_predictions_int.map(lambda x: float(x))
    #print("prediction:\n", predictions.take(1))
    lr_labels_predictions = validation_data_labels.zip(lr_predictions)
    #print(labels_predictions.take(3)) 
    #print("label prediction: \n", labels_predictions.first()) 
    #testErr = labels_predictions.filter(
    #    lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
    #print('Test Error = ' + str(testErr))

    #lr_metrics = BinaryClassificationMetrics(lr_labels_predictions)

    ### CONFUSION MATRIX
    # True positives, where label is 1 and prediction is 1
    tp_num = lr_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
    # True negatives, where label is 0 and prediction is 0
    tn_num = lr_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
    # FFalse positives, where label is 0 and prediction is 1
    fp_num = lr_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
    # False negatives, where  and label is 1 prediction is 0
    fn_num = lr_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

    precision = tp_num / (tp_num + fp_num)
    recall = tp_num / (tp_num + fn_num)

    f1 = 2 * (precision * recall) / (precision + recall)

    # true-positive rate
    # DO I NEED sensitivity?
    tpr =  tp_num / (tp_num + fn_num)
    # false-positive rate
    fpr = fp_num / (fp_num + tn_num)

    lr_grid[reg_param][iterations] = f1

# Best hyper-parameters
print(f"Logistoc Regression with LBGFS:")
best_hp = find_best_params(lr_grid)
print(f"Best hyper-parameters for LR(regParam, iterations) are {best_hp}")
lr_model = LogisticRegressionWithLBFGS.train(training_data,
                regParam=best_hp[0], iterations=best_hp[1])

# Evaluate model on test instances and compute test error
lr_predictions_int = lr_model.predict(test_data.map(lambda x: x.features))
lr_predictions = lr_predictions_int.map(lambda x: float(x))
#print("LR prediction:\n", lr_predictions.take(1))
lr_labels_predictions = test_data_labels.zip(lr_predictions)
#print(labels_predictions.take(3)) 
#print("label prediction: \n", lr_labels_predictions.first()) 

lr_metrics = BinaryClassificationMetrics(lr_labels_predictions)

### CONFUSION MATRIX
# True positives, where label is 1 and prediction is 1
tp_num = lr_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 1).count()
# True negatives, where label is 0 and prediction is 0
tn_num = lr_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 0).count()
# FFalse positives, where label is 0 and prediction is 1
fp_num = lr_labels_predictions.filter( lambda lp: lp[0] == 0 and lp[1] == 1).count()
# False negatives, where  and label is 1 prediction is 0
fn_num = lr_labels_predictions.filter( lambda lp: lp[0] == 1 and lp[1] == 0).count()

precision = tp_num / (tp_num + fp_num)
recall = tp_num / (tp_num + fn_num)

f1 = 2 * (precision * recall) / (precision + recall)

# true-positive rate
# DO I NEED sensitivity?
tpr =  tp_num / (tp_num + fn_num)
# false-positive rate
fpr = fp_num / (fp_num + tn_num)

print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"F1 score: {f1}")
print(f"Area Under ROC: {lr_metrics.areaUnderROC}")
print()
