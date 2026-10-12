import pandas as pd

# Display the first few rows of the DataFrame
def load_data(url):
    df = pd.read_csv(url)
    return df