import os
import pandas as pd

from sklearn.model_selection import train_test_split
from . import utils

def split_data(input_data: pd.DataFrame, config: dict, is_combine: bool):
    # Split predictor(s) and label
    X = input_data.drop(columns=config['label'])
    y = input_data[config['label']]

    # First splitting (train and test)
    X_train, X_test, \
        y_train, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=42,
            stratify=y_test
        )
    
    # Second splitting (test and valid)
    X_valid, X_test, \
        y_valid, y_test = train_test_split(
            X_test, y_test,
            test_size=config['valid_size'],
            random_state=42,
            stratify=y_test
        )
    
    if is_combine:
        train = pd.concat([X_train, y_train], axis=1)
        valid = pd.concat([X_valid, y_valid], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        return train, valid, test
    
    else:
        return X_train, X_valid, X_test, y_train, y_valid, y_test

if __name__ == '__main__':
    file_dir = os.path.dirname(__file__)

    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read raw dataset
    raw_dataset = pd.read_csv(
        os.path.join(file_dir, config['dataset_path']),
        delimiter=config['dataset_delimiter']
    )

    # 3. Split train, valid, and test set
    X_train, X_valid, X_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config, is_combine=False)
    
    train, valid, test = split_data(raw_dataset, config, is_combine=True)
    
    # 4. Dump data
    utils.pickle_dump(X_train, os.path.join(file_dir, config['train_path'][0]))
    utils.pickle_dump(y_train, os.path.join(file_dir, config['train_path'][1]))
    utils.pickle_dump(train, os.path.join(file_dir, config['train_path'][2]))

    utils.pickle_dump(X_valid, os.path.join(file_dir, config['valid_path'][0]))
    utils.pickle_dump(y_valid, os.path.join(file_dir, config['valid_path'][1]))
    utils.pickle_dump(valid, os.path.join(file_dir, config['valid_path'][2]))

    utils.pickle_dump(X_test, os.path.join(file_dir, config['test_path'][0]))
    utils.pickle_dump(y_test, os.path.join(file_dir, config['test_path'][1]))
    utils.pickle_dump(test, os.path.join(file_dir, config['test_path'][2]))