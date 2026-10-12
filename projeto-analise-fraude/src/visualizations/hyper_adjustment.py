from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def hyper_adjustment(X_train, y_train):

    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [50, 100]
    }

    grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss'),
        param_grid,
        scoring="recall",
        cv=3
    )

    grid.fit(X_train, y_train)

    return grid.best_params_