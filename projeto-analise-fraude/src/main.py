import pandas as pd
from data.data_loader import load_data
from sklearn.metrics import classification_report

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = load_data(url)

# Preprocessing - The dataset is preprocessed to prepare it for modeling
from preprocessing.prepare_fraud_data import prepare_variables_model
from methods.balancing_techniques.logistic_regression import logistic_regression

X_train, X_test, y_train, y_test, X, y = prepare_variables_model(df)
        
y_pred, model = logistic_regression(X_train, y_train, X_test)

metodo = 8

match metodo:
    case 1:
        # 1. Class Distribution - The dataset is highly imbalanced, with the majority of transactions being non-fraudulent.
        from methods.balancing_baseline.baseline import class_distribuition
        
        print("Class Distribution: ", class_distribuition(df))

    case 2:
        # 2. Logistic Regression Model - A simple logistic regression model is trained on the dataset to predict fraudulent transactions.
        from visualizations.viz_logistic import evaluate_logistic_model

        print("Logistic Regression Performance: ", evaluate_logistic_model(y_test, y_pred))

    case 3:
        from methods.balancing_techniques.pipeline import create_pipeline
        # pipeline
        
        y_pred_custom = create_pipeline(X_train, y_train, X_test, model)

        print("Pipeline Performance: ", classification_report(y_test, y_pred_custom))

    case 4:
        # 4. Tree decision Model - A decision tree model is trained on the dataset to predict fraudulent transactions.
        from methods.balancing_techniques.decision_tree import evaluate_tree
        y_pred_rf = evaluate_tree(df, X_train, y_train, X_test, X, y)
        print("Decision Tree Classifier Performance: ", classification_report(y_test, y_pred_rf))

    case 5:
        # 5. ROC Curve - The ROC curve is plotted to visualize the performance of the logistic regression model in distinguishing between fraudulent and non-fraudulent transactions.
        from visualizations.viz_roc_curve import evaluate_roc_curve
        evaluate_roc_curve(model, X_test, y_test)

    case 6:
        # 6. XGBoost Model - An XGBoost model is trained on the dataset to predict fraudulent transactions.
        from methods.balancing_techniques.xgb import evaluate_xgb
        y_pred_xgb, xgb = evaluate_xgb(X_train, y_train, X_test, y_test)
        print("XGBoost Classifier Performance: ", classification_report(y_test, y_pred_xgb))

    case 7:
        # feature importance - The feature importance of the XGBoost model is visualized to understand which features contribute the most to the model's predictions.
        from visualizations.plot_feature_importance import feature_importance
        # For graphics, recommended run in google colab or jupyter notebook
        # feature_importance(xgb)
    
    case 8:
        # hyperparameter adjustment - Testing different combinations for improve the model
        from visualizations.hyper_adjustment import hyper_adjustment
        best_params = hyper_adjustment(X_train, y_train)
        print("Best Parameters: ", best_params)

    case 9:
        # 9. explainability (SHAP) - The SHAP show how each feature contributes to the model's predictions, providing insights into the decision-making process of the model.
        from visualizations.shap import shap
        # For graphics, recommended run in google colab or jupyter notebook
        # shap(X_test, xgb)