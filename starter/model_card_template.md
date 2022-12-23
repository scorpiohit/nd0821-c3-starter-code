# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Scikit-Learn classifier called LogisticRegression is used with default paramters

## Intended Use
The intention is to to predict the salary of a customer using the features provided in raw data
 
## Training Data
Raw data can be obtained from : https://archive.ics.uci.edu/ml/datasets/census+income
This was later cleaned to remove trailing spaces. Only 80% was used for training.

## Evaluation Data
20% of data is used for evaluation

## Metrics
Model evaluation is done using below metrics score:
precision:  0.7171945701357466
recall:  0.6163318211276734
fbeta:  0.6629487626350645

## Ethical Considerations
Features like race and gender can create ethical bias. A slight discrimination can be noticed for female gender using recall metric.  

## Caveats and Recommendations
1. Need more demographic features
