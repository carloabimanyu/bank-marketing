import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from . import utils

def rus_fit_resample(X_train, y_train):
    return RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

def fit_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    
    return model

def eval_model(X_valid, y_valid, model):
    y_pred = model.predict(X_valid)
    score = recall_score(y_valid, y_pred)

    return score

def fit_eval_log(X_train, y_train, X_valid, y_valid, X_test, y_test, models: list):
    training_log = []
    for model in models:
        model = fit_model(X_train, y_train, model)
        log = {
            'model': model,
            'training_score': eval_model(X_train, y_train, model),
            'validation_score': eval_model(X_valid, y_valid, model),
            'test_score': eval_model(X_test, y_test, model),
        }
        training_log.append(log)

    with open('training_log.json', 'w') as json_file:
        json.dump(training_log, json_file, indent=4)

    return training_log

def select_model(training_log: list):
    for log in training_log:
        diff = abs(log['training_score'] - log['validation_score']) / log['training_score']
        if diff > 0.1:
            training_log.remove(log)

    selected_model = max(training_log, key=lambda model: (model['validation_score'] + model['test_score']) / 2)

    return selected_model['model']

if __name__ == '__main__':
    file_dir = os.path.dirname(__file__)

    # 1. Load config file
    config = utils.load_config()

    # 2. Load dataset
    X_train = utils.pickle_load(os.path.join(file_dir, config['train_processed_path'][0]))
    y_train = utils.pickle_load(os.path.join(file_dir, config['train_processed_path'][1]))

    X_valid = utils.pickle_load(os.path.join(file_dir, config['valid_processed_path'][0]))
    y_valid = utils.pickle_load(os.path.join(file_dir, config['valid_processed_path'][1]))

    X_test = utils.pickle_load(os.path.join(file_dir, config['test_processed_path'][0]))
    y_test = utils.pickle_load(os.path.join(file_dir, config['test_processed_path'][1]))

    # 3. Random under sampling train set
    X_rus, y_rus = rus_fit_resample(X_train, y_train)

    # 4. Fit, evaluate, and log models
    models = [LogisticRegression(), DecisionTreeClassifier()]
    training_log = fit_eval_log(X_train, y_train, X_valid, y_valid, X_test, y_test, models)

    # 5. Select model
    model = select_model(training_log)

    # 6. Dump model
    utils.pickle_dump(model, os.path.join(file_dir, config['model_path']))