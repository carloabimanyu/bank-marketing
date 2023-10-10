import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from . import util as utils

def load_dataset(config_data: dict):
    X_train = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["train_processed_path"][0]))
    y_train = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["train_processed_path"][1]))

    X_valid = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["valid_processed_path"][0]))
    y_valid = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["valid_processed_path"][1]))

    X_test = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["test_processed_path"][0]))
    y_test = utils.pickle_load(os.path.join(os.path.dirname(__file__), config_data["test_processed_path"][1]))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def rus_fit_resample(X, y):
    X_rus, y_rus = RandomUnderSampler(random_state=42).fit_resample(
        X, y
    )

    return X_rus, y_rus

def train_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_valid_pred = lr.predict(X_valid)
    y_test_pred = lr.predict(X_test)

    print("Result in valid dataset:")
    print(classification_report(y_valid, y_valid_pred))
    print("Result in test dataset:")
    print(classification_report(y_test, y_test_pred))

    return lr

if __name__ == "__main__":
    # 1. Load config file
    config = utils.load_config()

    # 2. Load dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset(config)

    # 3. Under sampling train set
    X_rus, y_rus = rus_fit_resample(X_train, y_train)

    # 4. Train model
    lr = train_model(X_rus, y_rus, X_valid, y_valid, X_test, y_test)

    # 5. Dump model
    utils.pickle_dump(lr, os.path.join(os.path.dirname(__file__), config["model_path"]))