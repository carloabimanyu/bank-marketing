import os
import uvicorn
import pandas as pd
import util as utils
import data_preparation
import preprocessing

from fastapi import FastAPI
from pydantic import BaseModel

config = utils.load_config()
model_data = utils.pickle_load(os.path.join(os.path.dirname(__file__), config["model_path"]))

class APIData(BaseModel):
    pass