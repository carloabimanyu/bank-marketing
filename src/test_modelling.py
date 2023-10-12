import os
import utils
import modelling

from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

file_dir = os.path.dirname(__file__)
config = utils.load_config()

X_train = utils.pickle_load(os.path.join(file_dir, config['train_processed_path'][0]))
y_train = utils.pickle_load(os.path.join(file_dir, config['train_processed_path'][1]))

X_valid = utils.pickle_load(os.path.join(file_dir, config['valid_processed_path'][0]))
y_valid = utils.pickle_load(os.path.join(file_dir, config['valid_processed_path'][1]))

X_test = utils.pickle_load(os.path.join(file_dir, config['test_processed_path'][0]))
y_test = utils.pickle_load(os.path.join(file_dir, config['test_processed_path'][1]))

def test_rus_fit_resample():
    # Arrange
    expected_proportions = 0.5

    # Act
    X_rus, y_rus = modelling.rus_fit_resample(X_train, y_train)

    # Assert
    assert y_rus.value_counts(normalize=True) == expected_proportions

def test_select_model():
    # Arrange
    expected_score = 0.75
    models = [LogisticRegression(), DecisionTreeClassifier()]

    # Act
    training_log = modelling.fit_eval_log(X_train, y_train, X_valid, y_valid, X_test, y_test, models)
    model = modelling.select_model(training_log)

    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Assert
    assert recall_score(y_valid, y_valid_pred) >= expected_score
    assert recall_score(y_test, y_test_pred) >= expected_score