# Guy Matz

# CSCI 49376 - Big Data

# Professor Xie

# Project 3 - Machine Learning

## Algorithms Chosen
Because the supplied data was labeled, using a supervised model made sense.
And, since the Label - Malignant / Benign - is "binary", using a
Classification model seemed like a good fit.

The MLLib package in PySpark provides a number of such models to
choose from.  I chose the following:

### Random Forest
The Random Forest algorithm is known to produce "good predictions that can
be easily understood".[^1]  One weakness Random Forest is its inability
to attribute findings to specific features provided in the training data.
However, because we are not - at this  point - looking
for any sort of correlation / causation, Random Forest is acceptable
for out purposes.

### Boosted Gradient Descent
Because Gradient Boosting is very similar to Random Forest, the same reasons
apply.  Gradient Boosting trees "can be more accurate than Random Forests",
though. [^2]  This only gives us more reason to use it, even though it
takes longer to run since this algorithm uses `boosting` instead of
`bagging`.

### Logistical Regression
Logistical Regression is a similar story.  While it, too, is appropriate for 
predicting a binary response, it works in a very different way than
the ensemble algorithms above.  Regardless, Logistic Regression


## Training Procedure
First the data was One-Hot Encoded to convert the Categorical Data - 
Malignant / Benign - into numerical data - 0 / 1 - for easier 
ingestion into our models.

Then, for each algorithm above, two hyper-parameters were chosen and given
a range of values for training.  Each model was then used to predict
Validation data and the best combination of hyper-parameters, i.e. the
pair that achieved the highest `f1`, were used against the Test Data.

The hyper-parameters and their values were (* = Default, ~ = Best):

- Random Forest
    - NumTrees
        1. 40
        1. 50
        1. 60 ~
    - maxDepth
        1. 3
        1. 4 *
        1. 5 ~
- Gradient Boosted
    - Learning Rate
        1. 0.09 ~
        1. 0.10 *
        1. 0.11
    - maxDepth
        1. 2 ~
        1. 3 *
        1. 4
- Logistic Regression with LBFGS
    - RegParam
        1. 0.09 ~
        1. 0.10 *
        1. 0.11
    - Number of Iterations
        1. 50 ~
        1. 100 *
        1. 150

## Data Splits

Data was split into 3 groupings: Training, Validation & Test.  Training data
- 60% - was used to train each model with the various hyper-parameters
above, using the Validation data - 20% - to determine the best combination of
hyper-parameters.  Test data  - 20% - was used post-validation to determine
Precision, Recall, F1 & AUR.

## Test Results
#### Random Forest (numTrees=75, maxDepth=30)
```
Precision: 0.98
Recall: 0.9423076923076923
F1 score: 0.9607843137254902
Area Under ROC: 0.9645762711864407
```

#### Gradient Boosted (learningRate=0.09, maxDepth=3)
```
Precision: 0.8809523809523809
Recall: 0.9487179487179487
F1 score: 0.9135802469135802
Area Under ROC: 0.9261904761904761
```

#### Logistic Regression (regParam=0.1, iterations=50)
```
Precision: 0.9473684210526315
Recall: 0.9230769230769231
F1 score: 0.935064935064935
Area Under ROC: 0.9534139402560454
```

### Comparison of Algorithms.
All of the algorithms did an admirable job!  I am sure I could not do any
better on my own!  Somewhat surprisingly, Random Forest performed the best.
I would have expected Gradient Boosted to have performed better, since it
is "serial" and takes longer to train the model.

## Discussion

### Limitations and future improvement.
One huge limitation - as I see it - is that our models do not attempt to
explain anything.  We can make predictions based on our models, but we have
no idea why those predictions are made.  Perhaps other models would be
useful for this.  One future improvement I would like to make to my code
is a complete refactorization.  I ended up copy/pasting large pieces of 
code due to my inability to get things working initially.  I discovered 
a bit late in the day that I needed to convert `int`s to `float`s in my 
`prediction` RDDs in order to have them "zippabble"

[^1]: https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/
[^2]: https://www.baeldung.com/cs/gradient-boosting-trees-vs-random-forests
