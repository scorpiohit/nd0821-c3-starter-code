from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    return LogisticRegression().fit(X_train, y_train)



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def compute_metrics_by_slice(model,
                             df,
                             cat_columns,
                             target,
                             training=False,
                             encoder,
                             lb):
    """
    This function outputs the performance of the model on slices of the data
    args:
        - model : Trained model binary file
        - df : Input dataframe
        - cat_columns : list of categorical columns
        - target : Class label  string
        - training : Boolean to flag training
        - encoder : fitted One Hot Encoder
        - lb : label binarizer
    returns:
        - metrics : Output dataframe containing metric
    """

    rows_list = list()
    for col in cat_columns:
        for category in df[col].unique():
            row = {}
            df_tmp = df[df[col] == category]

            x, y, _, _ = process_data(
                X=df_tmp,
                categorical_features=cat_columns,
                label=target,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, x)
            precision, recall, f_one = compute_model_metrics(y, preds)

            row['col'] = col
            row['category'] = category
            row['precision'] = precision
            row['recall'] = recall
            row['f1'] = f_one

            rows_list.append(row)

    metrics = pd.DataFrame(rows_list, columns=["col", "category", "precision", "recall", "f1"])

    return metrics
