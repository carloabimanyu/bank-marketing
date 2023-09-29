import os
import uvicorn
import numpy as np
import pandas as pd
import util as utils
import data_preparation
import preprocessing

from fastapi import FastAPI
from pydantic import BaseModel

config = utils.load_config()
model_data = utils.pickle_load(os.path.join(os.path.dirname(__file__), config["model_path"]))
train_columns = utils.pickle_load(os.path.join(os.path.dirname(__file__), config["train_processed_path"][0])).columns

class APIData(BaseModel):
    age: int
    job: object
    marital: object
    education: object
    default: object
    balance: int
    housing: object
    loan: object
    contact: object
    day: int
    month: object
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: object

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: APIData):
    # Convert API data to DataFrame
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # Convert dtype
    data = pd.concat(
        [
            data[config["int_columns"]].astype(np.int64),
            data.drop(columns=config["int_columns"], axis=1)
        ],
        axis=1
    )
    
    # Preprocessing and feature engineering API data
    data_processed = preprocessing.preprocessing(data, is_api=True)
    data_processed = preprocessing.feature_engineering(data_processed)

    # Check data range
    try:
        data_preparation.check_data(data, config)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Predict data
    data_processed.columns = train_columns
    y_pred = model_data.predict(data_processed)

    if y_pred[0] == 0:
        y_pred = "Client will not subscribe a term deposit."
    else:
        y_pred = "Client will subscribe a term deposit."

    return {"res": y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8080)