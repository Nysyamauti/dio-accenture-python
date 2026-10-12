def class_distribuition(df):
    """
    Calculate the class distribution of the 'Class' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Class' column.

    Returns:
    pd.Series: A Series containing the normalized value counts of each class.
    """
    return df['Class'].value_counts(normalize=True)