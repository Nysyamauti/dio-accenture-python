import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_variables_model(df):
    """
    Prepares the fraud dataset for modeling.

    Returns:
    X_train, X_test, y_train, y_test: Split and preprocessed features and target variables.
    """
    # 1. Log transformation of the Amount feature
    df['Amount_log'] = np.log1p(df["Amount"])

    # 2. Standardize the Amount_log feature
    scaler = StandardScaler()
    df[['Amount_scaled']] = scaler.fit_transform(df[["Amount"]])

    # 3. Define features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 4. Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, X, y