# Put the code for your API here.
from fastapi import FastAPI
from typing import Union
import joblib
import pandas as pd

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import os
import sys

#file_dir = os.path.dirname(os.path.abspath("__file__"))
#sys.path.insert(0, file_dir)
sys.path.insert(0, "starter/starter")

from ml.data import process_data
from ml.model import inference

# To give Heroku the ability to pull in data from DVC upon app start up
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class dataInput(BaseModel):
    age: int = Field(..., example=35)
    workclass: str = Field(..., example="Federal-gov")
    fnlgt: int = Field(..., example=249409)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num", example=9)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Other-service")
    relationship: str = Field(..., example="Own-child")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )


# Instantiate the app.
app = FastAPI()

model = joblib.load(os.path.join('./starter/model/', 'model.pkl'))
encoder = joblib.load( os.path.join('./starter/model/', 'encoder.pkl'))
lb = joblib.load(os.path.join('./starter/model/', 'lb.pkl'))

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

#Define a POST on specific endpoint.
@app.post("/inference/")
async def predict(item: dataInput):
    cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    X = pd.DataFrame(data = [item.dict(by_alias=True)], index = [0])
    X,_,_,_ = process_data(X, cat_features, label=None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)

    if pred[0]:
        pred = {'salary': '>50k'}
    else:
        pred = {'salary': '<=50k'}
    return pred
