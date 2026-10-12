import pandas as pd
from imblearn.over_sampling import SMOTE

def evalueate_data_balancing(df):
    # Undersampling
    fraudes = df[df['Class'] == 1]
    normais = df[df['Class'] == 0].sample(n=len(fraudes), random_state=42)

    df_undersampled = pd.concat([fraudes, normais])

    # Oversampling
    smote = SMOTE()

    X_res, y_res = smote.fit_resample(X, y)