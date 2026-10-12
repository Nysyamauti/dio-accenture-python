from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from methods.balancing_techniques.logistic_regression import logistic_regression
""" 
5. Pipeline: 
    A pipeline is created to streamline the preprocessing and modeling steps, 
    allowing for easier experimentation with different models and techniques.
"""

def create_pipeline(X_train, y_train, X_test, model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", logistic_regression(X_train, y_train, X_test)[1])
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    threshold = 0.3

    y_probs = model.predict_proba(X_test)[:,1]

    y_pred_custom = (y_probs >= threshold).astype(int)

    return y_pred_custom