import os
import pandas as pd
import util as utils
import copy

from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(os.path.join(os.path.dirname(__file__), config["dataset_path"]), \
                       delimiter=config["dataset_delimiter"])

def check_data(input_data: pd.DataFrame, config: dict):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    # Check column data types
    assert input_data.select_dtypes("int").columns.to_list() == \
        config["int_columns"], "an error occurs in int column(s)."
        
    # Check range of age
    assert input_data[config["int_columns"][0]].between(
        config["range_age"][0],
        config["range_age"][1]
    ).sum() == len(input_data), "an error occurs in range_age."

    # Check range of balance
    assert input_data[config["int_columns"][1]].between(
        config["range_balance"][0],
        config["range_balance"][1]
    ).sum() == len(input_data), "an error occurs in range_balance."

    # Check range of day
    assert input_data[config["int_columns"][2]].between(
        config["range_day"][0],
        config["range_day"][1]
    ).sum() == len(input_data), "an error occurs in range_day."

    # Check range of duration
    assert input_data[config["int_columns"][3]].between(
        config["range_duration"][0],
        config["range_duration"][1]
    ).sum() == len(input_data), "an error occurs in range_duration."

    # Check range of campaign
    assert input_data[config["int_columns"][4]].between(
        config["range_campaign"][0],
        config["range_campaign"][1]
    ).sum() == len(input_data), "an error occurs in range_campaign."

    # Check range of pdays
    assert input_data[config["int_columns"][5]].between(
        config["range_pdays"][0],
        config["range_pdays"][1]
    ).sum() == len(input_data), "an error occurs in range_pdays."

    # Check range of previous
    assert input_data[config["int_columns"][6]].between(
        config["range_previous"][0],
        config["range_previous"][1]
    ).sum() == len(input_data), "an error occurs in range_previous."

def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor(s) and label
    X = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=42,
        stratify=y
    )

    # 2nd split test and valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size=config["valid_size"],
        random_state=42,
        stratify=y_test
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Data defense for non API data
    check_data(raw_dataset, config)

    # 4. Splitting train, valid, and test set
    X_train, X_valid, X_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)
    
    # 5. Save train, valid, and test set
    utils.pickle_dump(X_train, os.path.join(os.path.dirname(__file__), config["train_path"][0]))
    utils.pickle_dump(y_train, os.path.join(os.path.dirname(__file__), config["train_path"][1]))

    utils.pickle_dump(X_valid, os.path.join(os.path.dirname(__file__), config["valid_path"][0]))
    utils.pickle_dump(y_valid, os.path.join(os.path.dirname(__file__), config["valid_path"][1]))

    utils.pickle_dump(X_test, os.path.join(os.path.dirname(__file__), config["test_path"][0]))
    utils.pickle_dump(y_test, os.path.join(os.path.dirname(__file__), config["test_path"][1]))