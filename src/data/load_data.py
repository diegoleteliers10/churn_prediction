import pandas as pd

def load_data():

  print("Loading data...")
  df = pd.read_csv('../../data/raw/Telco-Customer-Churn.csv')

  if df.empty:
    raise ValueError("No data found")

  print("Data loaded successfully")

  return df






