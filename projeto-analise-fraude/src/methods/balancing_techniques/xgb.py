from xgboost import XGBClassifier

def evaluate_xgb(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier(
        scale_pos_weight=10, # helps on imbalanced datasets
        use_label_encoder=False,
        eval_metric='logloss',
    )

    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)

    return y_pred_xgb, xgb
