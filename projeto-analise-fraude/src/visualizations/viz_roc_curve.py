from methods.balancing_techniques.roc_curve import plot_roc_curve
def evaluate_roc_curve(model, X_test, y_test):
    print("Evaluating ROC Curve...")
    print(plot_roc_curve(model, X_test, y_test))