we have labeled data, amd we want to build a model that is able to 
classify new data as one of two groups.  For these reasons we will
use Classification models.

# Algorithms Chosen
Because the supplied data was labeled, using a supervised model made sense.
And, since the Label - Malignant / Benign - is "binary", using a
Classification model seemed like a good fit.

The MLLib package in PySpark provides a number of such models to
choose from.  I chose the following:

## Random Forest
The Random Forest algorithm is known to produce "good predictions that can
be easily understood".[^1]  One weakness Random Forest is its inability
to attribute findings to specific features provided in the training data.
However, because we are not - at this  point - looking
for any sort of correlation / causation, Random Forest is acceptable
for out purposes.

## Boosted Gradient Descent
Because Gradient Boosting is very similar to Random Forest, the same reasons
apply.  Gradient Boosting trees "can be more accurate than Random Forests",
though. [^2]  This only gives us more reason to use it.

## Logistical Regression
Logistical Regression is a similar story.  While it, too, is appropriate for 
predicting a binary response, it works in a very different way than
the ensemble algorithms above.  Regardless, Logistic Regression


# Training Procedure
For each algorithm above, two hyper-parameters were chosen and given a range
of values for training.  Each model was then used to predict Validation data
and the best combination of hyper-parameters, i.e. the pair that 
achieved the highest `f1`, were used against the Test Data.

The hyper-parameters and their values were (* = Default, ~ = Best):
- Random Forest
  - Feature Subset Strategy
    1. auto
    1. sqrt ~
    1. log2
    1. onethird
  - Impurity
    1. gini *~
    1. entropy
- Gradient Boosted
  - Loss Function
    1. logLoss *
    1. leastSquaresError ~
    1. leastAbsoluteError
  - Number of Iterations
    1. 50 ~
    1. 100 *
    1. 150
- Logistic Regression with LBFGS
  - Regulizer Type
    1. l1 ~
    1. l2 *
  - Number of Iterations
    1. 50 ~
    1. 100 *
    1. 150

# Data Splits

Data was split into 3 groupings: Training, Validation & Test.  Training data
- 60% - was used to train each model with the various hyper-parameters
above, using the Validation data - 20% - to determine the best combination of
hyper-parameters.  Test data  - 20% - was used post-validation to determine
Precision, Recall, F1 & AUR.

# Testing Results
## Random Forest (featureSubsetStrategy=sqrt, Impurity=gini)
Precision: 0.9361702127659575
Recall: 0.8979591836734694
F1 score: 0.9166666666666666
Area Under ROC: 0.9328738387773449

## GRADIENT BOOSTED (loss=leastSquareError, numIterations=50)
Precision: 0.9565217391304348
Recall: 0.8979591836734694
F1 score: 0.9263157894736843
Area Under ROC: 0.9435386473429952

## Logistic Regression (regType=l1, numIterations=50)
Precision: 0.9565217391304348
Recall: 0.8979591836734694
F1 score: 0.9263157894736843
Area Under ROC: 0.9435386473429952

• Comparison of two algorithms.
– Discussion
• Limitations and future improvement.

[^1]: https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/
[^2]: https://www.baeldung.com/cs/gradient-boosting-trees-vs-random-forests
