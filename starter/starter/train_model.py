# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import logging
import sys
import os
import numpy as np

file_dir = os.path.dirname(os.path.abspath("__file__"))
print(file_dir)
sys.path.insert(0, file_dir)

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_metrics_by_slice

# Add code to load in the data.
data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
   )

with open('../model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('../model/lb.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Train and save a model.
model = train_model(X_train, y_train)

with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Getting the predictions and metrics score
logging.info('Computing the model metric...')
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info('printing the model performance as txt...')
print ("precision: ", precision) 
print ("recall: ", recall)
print ("fbeta: ", fbeta)

# Compute the model performance on slices
logging.info('computing the model performance on slices...')
performance = compute_metrics_by_slice(model, test, cat_columns=cat_features, target="salary", training=False, encoder=encoder, lb=lb)
logging.info('Saving the model performance on slices as txt...')
saved_performance = performance.to_numpy()
np.savetxt('../model/slice_output.txt', saved_performance, fmt='%s')
