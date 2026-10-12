import matplotlib.pyplot as plt

def feature_importance(xgb):
    importancias = xgb.feature_importances_

    plt.bar(range(len(importancias)), importancias)
    plt.title("Feature Importance")
    return plt.show()