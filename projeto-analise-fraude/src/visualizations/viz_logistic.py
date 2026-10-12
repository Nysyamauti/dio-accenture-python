from sklearn.metrics import classification_report

def evaluate_logistic_model(y_test, y_pred):
    print("\nLogistic Regression Model Evaluation:")
    print(classification_report(y_test, y_pred))