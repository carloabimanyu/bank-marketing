import yaml
import pandas as pd

from typing import NamedTuple, Union
from sklearn.base import ClassifierMixin, RegressorMixin
    
class DataPreparationOutputs(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    
def data_preparation() -> DataPreparationOutputs:
    # 0. Import module
    import src
    
    # 1. Load config file
    with open('/app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # 2. Load raw dataset
    raw_dataset = pd.read_csv(f'{config["BUCKET"]}{config["dataset_path"]}', delimiter=config['dataset_delimiter'])
    
    # 3. Data defense for non API data
    src.data_preparation.check_data(raw_dataset, config)
    
    # 4. Splitting train, valid, and test set
    X_train, X_valid, X_test, \
        y_train, y_valid, y_test = src.data_preparation.split_data(raw_dataset, config)
    
    # 5. Concat train, valid, and test set
    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    return DataPreparationOutputs(train=train, valid=valid, test=test)

class DataPreprocessingOutputs(NamedTuple):
    train_processed: pd.DataFrame
    valid_processed: pd.DataFrame
    test_processed: pd.DataFrame
    
def preprocessing(data_preparation_outputs: DataPreparationOutputs) -> DataPreprocessingOutputs:
    # 0. Import module
    import src
    
    # 1. Retrieve data from previous component
    train = data_preparation_outputs.train
    valid = data_preparation_outputs.valid
    test = data_preparation_outputs.test
    
    # 2. Preprocess data
    train_processed = src.preprocessing.preprocessing(train, is_api=False)
    valid_processed = src.preprocessing.preprocessing(valid, is_api=False)
    test_processed = src.preprocessing.preprocessing(test, is_api=False)
    
    # 3. Feature engineering
    train_processed = src.preprocessing.feature_engineering(train_processed)
    valid_processed = src.preprocessing.feature_engineering(valid_processed)
    test_processed = src.preprocessing.feature_engineering(test_processed)
    
    return DataPreprocessingOutputs(
        train_processed=train_processed,
        valid_processed=valid_processed,
        test_processed=test_processed
    )

class ModellingOutputs(NamedTuple):
    model: Union[ClassifierMixin, RegressorMixin]
    
def modelling(data_preprocessing_outputs: DataPreprocessingOutputs) -> ModellingOutputs:
    # 0. Import module
    import src
    
    # 1. Load config file
    with open('/app/config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    # 2. Retrieve data from previous component
    X_train = data_preprocessing_outputs.train_processed.drop(columns=config['label'])
    y_train = data_preprocessing_outputs.train_processed[config['label']]
    
    X_valid = data_preprocessing_outputs.valid_processed.drop(columns=config['label'])
    y_valid = data_preprocessing_outputs.valid_processed[config['label']]
    
    X_test = data_preprocessing_outputs.test_processed.drop(columns=config['label'])
    y_test = data_preprocessing_outputs.test_processed[config['label']]
    
    # 3. Random under sampling train set
    X_rus, y_rus = src.modelling.rus_fit_resample(X_train, y_train)
    
    # 4. Train model
    model = src.modelling.train_model(X_rus, y_rus, X_valid, y_valid, X_test, y_test)
    
    return ModellingOutputs(model=model)