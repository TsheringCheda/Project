# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Function to read CSV with error handling
def read_csv_with_error_handling(filepath):
  try:
    dataset = pd.read_csv(filepath, header=None)
  except FileNotFoundError:
    st.error("Error: 'bhutan_tourism_data.csv' file not found. Please ensure the file exists and is accessible.")
    return None

  return dataset

# Function for data analysis and visualization
def analyze_data(dataset):
  # Data cleaning
  data = dataset.iloc[:, 2:]  # Select columns from 2nd to last (excluding first 2)
  data.drop([10, 11, 12, 13, 14], inplace=True)  # Drop specific rows

  # Data transformation
  data = data.T  # Transpose
  data.drop(2, inplace=True)  # Drop specific row after transpose
  data.drop(columns=[1, 2, 3, 4, 6, 7, 8, 9, 15], inplace=True)  # Drop specific columns

  # Set column names
  data_cols = ['Years', 'Number of Tourists']
  data.columns = data_cols

  # **Improved Index Setting (considering potential mismatch):**
  num_rows = len(data)
  if num_rows != 25:  # Adjust based on your actual data
    data.index = range(1991, 1991 + num_rows)  # Set index based on actual number of rows
  else:
    data.index = range(1991, 2016)  # Use original range if lengths match

  # Extract year as integer
  data['Years'] = data['Years'].str.extract(r'(\d{4})').fillna(-1).astype(int)

  # Ensure 'Number of Tourists' is numeric (handling potential errors)
  data['Number of Tourists'] = pd.to_numeric(data['Number of Tourists'], errors='coerce')

  # Handle NaN values in 'Number of Tourists' (adjust strategy as needed)
  data['Number of Tourists'].fillna(data['Number of Tourists'].mean(), inplace=True)

  # Plot the time series
  data.plot.line(x='Years', y='Number of Tourists')
  plt.title('Bhutan Tourist Arrivals (1991-2015)')
  plt.xlabel('Years')
  plt.ylabel('Number of Tourists')
  plt.grid(True)
  plt.show()

  # Test for stationarity
  result = adfuller(data['Number of Tourists'])
  st.write("ADF Test Results:")
  st.write(f"ADF Test Statistic: {result[0]:.4f}")
  st.write(f"p-value: {result[1]:.4f}")
  st.write(f"# Lags Used: {result[2]}")
  st.write(f"Number of Observations Used: {result[3]}")

  if result[1] <= 0.05:
    st.write("Strong evidence against the null hypothesis(H0), reject the null hypothesis. Data has no unit root and is stationary")
  else:
    st.write("Weak Evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

# Create the Streamlit app
def main():
  st.title("Bhutan Tourism Data Analysis")

  # Upload the CSV file
  uploaded_file = st.file_uploader("Upload Bhutan Tourism Data (CSV)")

  if uploaded_file is not None:
    dataset = read_csv_with_error_handling(uploaded_file)
    if dataset is not None:
      analyze_data(dataset)
    else:
      st.error("Error reading CSV file. Please check the file format and encoding.")

if __name__ == "__main__":
  main()