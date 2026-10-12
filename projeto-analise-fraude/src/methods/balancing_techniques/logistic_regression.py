from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, y_train, X_test):
    """
    Trains a Logistic Regression model and makes predictions on the test set.

    Parameters:
    X_train: Training features
    y_train: Training target variable
    X_test: Testing features

    Returns:
    y_pred: Predicted values for the test set
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model