import util as utils
import modelling

from sklearn.metrics import recall_score

def test_load_dataset():
    # Arrange
    config = utils.load_config()
    expected_predictors = config["predictors"]

    # Act
    X_train, X_valid, X_test, y_train, y_valid, y_test = modelling.load_dataset(config)

    # Assert
    assert X_train.columns.tolist() == expected_predictors
    assert X_valid.columns.tolist() == expected_predictors
    assert X_test.columns.tolist() == expected_predictors

def test_rus_fit_resample():
    # Arrange
    config = utils.load_config()
    expected_proportions = 0.5

    # Act
    X_train, X_valid, X_test, y_train, y_valid, y_test = modelling.load_dataset(config)
    X_rus, y_rus = modelling.rus_fit_resample(X_train, y_train)

    # Assert
    assert y_rus.value_counts(normalize=True)[0] == expected_proportions

def test_train_model():
    # Arrange
    config = utils.load_config()
    expected_recall = 0.75

    # Act
    X_train, X_valid, X_test, y_train, y_valid, y_test = modelling.load_dataset(config)
    X_rus, y_rus = modelling.rus_fit_resample(X_train, y_train)

    lr = modelling.train_model(X_rus, y_rus, X_valid, y_valid, X_test, y_test)
    y_valid_pred = lr.predict(X_valid)
    y_test_pred = lr.predict(X_test)

    # Assert
    assert recall_score(y_valid, y_valid_pred) >= expected_recall
    assert recall_score(y_test, y_test_pred) >= expected_recall