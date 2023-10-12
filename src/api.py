import os
import uvicorn
import numpy as np
import pandas as pd
import utils
import data_validation
import data_preprocessing

from fastapi import FastAPI
from pydantic import BaseModel

file_dir = os.path.dirname(__file__)
config = utils.load_config()
model = utils.pickle_load(os.path.join(file_dir, config['model_path']))
train = utils.pickle_load(os.path.join(file_dir, config["train_processed_path"][0]))

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

@app.get('/')
def home():
    return 'Hello, FastAPI up!'

@app.post('/predict/')
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
    data_processed = data_preprocessing.preprocessing(data, config, is_api=True)
    data_processed = data_preprocessing.feature_engineering(data_processed, config, train)

    # Check data range
    try:
        data_validation.check_data_range(data, config)
    except AssertionError as ae:
        return {
            'res': [],
            'error_msg': str(ae)
        }
    
    # Predict data
    data_processed.columns = train.columns
    y_pred = model.predict(data_processed)

    if y_pred[0] == 0:
        y_pred = 'Client will not subscribe a term deposit.'
    else:
        y_pred = 'Client will subscribe a term deposit.'

    return {
        'res': y_pred,
        'error_msg': ''
    }

if __name__ == '__main__':
    uvicorn.run('api:app', host='127.0.0.1', port=8080)