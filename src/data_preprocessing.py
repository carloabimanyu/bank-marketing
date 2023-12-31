import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from . import utils

def preprocessing(data, config: dict, is_api: bool):
    # 1. Re-categorize job
    data['job'] = data['job'].apply(lambda x: x if x in config['job_new_category']['keep'] else config['job_new_category']['replace'])

    # 2. Replace unknown to secondary for education column
    data['education'] = data['education'].apply(lambda x: config['education_replace']['to'] if x in config['education_replace']['from'] else x)

    # 3. Merge cellular and telephone for contact column
    data['contact'] = data['contact'].apply(lambda x: config['contact_merge']['to'] if x in config['contact_merge']['from'] else x)

    # 4. Replace -1 to 999 for pdays column
    data['pdays'] = data['pdays'].apply(lambda x: config['pdays_replace']['to'] if x == config['pdays_replace']['from'] else x)

    # 5. Label encoding
    if is_api:
        columns = config['label_encoding_columns']
        columns.remove('y')
    else:
        columns = config['label_encoding_columns']

    for column in columns:
        data[column] = data[column].apply(lambda x: config[f'{column}_code'][x])
    
    return data

def feature_engineering(data, config: dict, train):
    # 1. One-Hot encoding
    for column in config['one_hot_encoding_columns']:
        encoder = OneHotEncoder(drop='first', handle_unknown='error')
        encoder.fit(preprocessing(train, is_api=False)[[column]])
        encoded_data = encoder.transform(data[[column]])
        encoded_df = pd.DataFrame(encoded_data.toarray(),
                                  columns=encoder.get_feature_names_out([column]),
                                  index=data.index)
        data = pd.concat([data, encoded_df], axis=1)
        data = data.drop(columns=[column])

    return data

if __name__ == '__main__':
    file_dir = os.path.dirname(__file__)

    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read dataset
    X_train = utils.pickle_load(os.path.join(file_dir, config['train_path'][0]))
    y_train = utils.pickle_load(os.path.join(file_dir, config['train_path'][1]))
    train = pd.concat([X_train, y_train], axis=1)

    X_valid = utils.pickle_load(os.path.join(file_dir, config['valid_path'][0]))
    y_valid = utils.pickle_load(os.path.join(file_dir, config['valid_path'][1]))
    valid = pd.concat([X_valid, y_valid], axis=1)

    X_test = utils.pickle_load(os.path.join(file_dir, config['test_path'][0]))
    y_test = utils.pickle_load(os.path.join(file_dir, config['test_path'][1]))
    test = pd.concat([X_test, y_test], axis=1)

    # 3. Preprocessing
    train_processed = preprocessing(train, is_api=False)
    valid_processed = preprocessing(valid, is_api=False)
    test_processed = preprocessing(test, is_api=False)

    # 4. Feature engineering
    train_processed_feng = feature_engineering(train_processed)
    valid_processed_feng = feature_engineering(valid_processed)
    test_processed_feng = feature_engineering(test_processed)

    # 5. Dump data
    utils.pickle_dump(
            train_processed.drop(columns=[config['label']]),
            os.path.join(file_dir, config['train_processed_path'][0])
    )
    utils.pickle_dump(
            train_processed[config['label']],
            os.path.join(file_dir, config['train_processed_path'][1])
    )
    utils.pickle_dump(
            train_processed,
            os.path.join(file_dir, config['train_processed_path'][2])
    )

    utils.pickle_dump(
            valid_processed.drop(columns=[config['label']]),
            os.path.join(file_dir, config['valid_processed_path'][0])
    )
    utils.pickle_dump(
            valid_processed[config['label']],
            os.path.join(file_dir, config['valid_processed_path'][1])
    )
    utils.pickle_dump(
            valid_processed,
            os.path.join(file_dir, config['valid_processed_path'][2])
    )

    utils.pickle_dump(
            test_processed.drop(columns=[config['label']]),
            os.path.join(file_dir, config['test_processed_path'][0])
    )
    utils.pickle_dump(
            test_processed[config['label']],
            os.path.join(file_dir, config['test_processed_path'][1])
    )
    utils.pickle_dump(
            test_processed,
            os.path.join(file_dir, config['test_processed_path'][2])
    )