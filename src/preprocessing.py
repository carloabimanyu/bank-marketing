import os
import pandas as pd
import util as utils

from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict):
    X_train = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["train_path"][0]))
    y_train = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["train_path"][1]))

    X_valid = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["valid_path"][0]))
    y_valid = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["valid_path"][1]))

    X_test = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["test_path"][0]))
    y_test = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["test_path"][1]))

    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, valid, test

def preprocessing(data):
    # 0. Copy data
    data = data.copy()

    # 1. Re-categorize job
    data["job"] = data["job"].apply(lambda x: x if x in config["job_new_category"]["keep"] else config["job_new_category"]["replace"])

    # 2. Replace unknown to secondary for education column
    data["education"] = data["education"].apply(lambda x: config["education_replace"]["to"] if x in config["education_replace"]["from"] else x)

    # 3. Merge cellular and telephone for contact column
    data["contact"] = data["contact"].apply(lambda x: config["contact_merge"]["to"] if x in config["contact_merge"]["from"] else x)

    # 4. Replace -1 to 999 for pdays column
    data["pdays"] = data["pdays"].apply(lambda x: config["pdays_replace"]["to"] if x == config["pdays_replace"]["from"] else x)

    # 5. Label encoding
    for column in config["label_encoding_columns"]:
        data[column] = data[column].apply(lambda x: config[f"{column}_code"][x])
    
    return data

def feature_engineering(data):
    # 0. Copy data
    data = data.copy()

    # 1. One-Hot encoding
    for column in config["one_hot_encoding_columns"]:
        encoder = OneHotEncoder(drop="first", handle_unknown="error")
        encoder.fit(preprocessing(train)[[column]])
        encoded_data = encoder.transform(data[[column]])
        encoded_df = pd.DataFrame(encoded_data.toarray(),
                                  columns=encoder.get_feature_names_out([column]),
                                  index=data.index)
        data = pd.concat([data, encoded_df], axis=1)
        data = data.drop(columns=[column])

    return data

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    train, valid, test = load_dataset(config)

    # 3. Preprocessing dataset
    train_processed = preprocessing(train)
    valid_processed = preprocessing(valid)
    test_processed = preprocessing(test)

    # 4. Feature engineering
    train_processed = feature_engineering(train_processed)
    valid_processed = feature_engineering(valid_processed)
    test_processed = feature_engineering(test_processed)

    # 5. Dump dataset
    utils.pickle_dump(
            train_processed.drop(columns=[config["label"]]),
            os.path.join(os.path.dirname(__file__), config["train_processed_path"][0])
    )
    utils.pickle_dump(
            train_processed[config["label"]],
            os.path.join(os.path.dirname(__file__), config["train_processed_path"][1])
    )

    utils.pickle_dump(
            valid_processed.drop(columns=[config["label"]]),
            os.path.join(os.path.dirname(__file__), config["valid_processed_path"][0])
    )
    utils.pickle_dump(
            valid_processed[config["label"]],
            os.path.join(os.path.dirname(__file__), config["valid_processed_path"][1])
    )

    utils.pickle_dump(
            test_processed.drop(columns=[config["label"]]),
            os.path.join(os.path.dirname(__file__), config["test_processed_path"][0])
    )
    utils.pickle_dump(
            test_processed[config["label"]],
            os.path.join(os.path.dirname(__file__), config["test_processed_path"][1])
    )