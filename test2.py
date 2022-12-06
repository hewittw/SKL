# need to fix sci kit learn install?? maybe just in visual studio code 
# learn how to go through and do what i want to with my flight times data set
# delete flight times data set when done

# fix github - get dads help with this


#--------------------------------------------------------------------------------#
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

pd.set_option("display.max_columns", 500)
plt.style.use("seaborn-colorblind")
pal = sns.color_palette()

#--------------------------------------------------------------------------------#
# Read in Data 


raw_df = pd.read_csv('flightsData.csv') # Ask about df naming conventions
print(raw_df)
# df_subset_0 = create_df_subset(raw_df, 0)
# df_subset_1 = create_df_subset(raw_df, 1)




# parquet_files = glob("../input/flight-delay-dataset-20182022/*.parquet")
# column_subset = [
#     "FlightDate",
#     "Airline",
#     "Flight_Number_Marketing_Airline",
#     "Origin",
#     "Dest",
#     "Cancelled",
#     "Diverted",
#     "CRSDepTime",
#     "DepTime",
#     "DepDelayMinutes",
#     "OriginAirportID",
#     "OriginCityName",
#     "OriginStateName",
#     "DestAirportID",
#     "DestCityName",
#     "DestStateName",
#     "TaxiOut",
#     "TaxiIn",
#     "CRSArrTime",
#     "ArrTime",
#     "ArrDelayMinutes",
# ]

# print(column_subset)


# dfs = []
# for f in parquet_files:
#     dfs.append(pd.read_parquet(f, columns=column_subset))
# df = pd.concat(dfs).reset_index(drop=True)

# cat_cols = ["Airline", "Origin", "Dest", "OriginStateName", "DestStateName"]
# for c in cat_cols:
#     df[c] = df[c].astype("category")