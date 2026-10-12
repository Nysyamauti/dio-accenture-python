def shap(X_test, xgb):
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_test)

    shap.plots.bar(shap_values)