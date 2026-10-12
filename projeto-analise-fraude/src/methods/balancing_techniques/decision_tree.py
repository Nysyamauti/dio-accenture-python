import pandas as pd
from imblearn.over_sampling import SMOTE

def evaluate_tree(df, X_train, y_train, X_test, X, y):
    # Undersampling
    fraudes = df[df['Class'] == 1]
    normais = df[df['Class'] == 0].sample(n=len(fraudes), random_state=42)

    df_undersampled = pd.concat([fraudes, normais])

    # Oversampling
    smote = SMOTE()

    X_res, y_res = smote.fit_resample(X, y)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    return y_pred_rf