from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    #plt.show() - Uncomment this line if you want to display the plot
    return 
