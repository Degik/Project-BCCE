import torch
import datetime
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler

FEAT_COLS = ['Close']

def importDataset(file_name:str) -> pd.DataFrame:
    dataset = []
    try:
        dataset = pd.read_csv(file_name, header=None, sep=",", skiprows=1)
    except Exception as e:
        print("Error | Can not read dataset cup for take input")
        exit(1)
    return dataset[1]



def get_9_business_days_back(today:str):
    business_day_20_back = pd.to_datetime(today) - BDay(9)
    return business_day_20_back.strftime('%Y-%m-%d')

# Normalize the data
# All data will be convert inside the range [0, 1]
def normalizeData(train_set, val_set):
    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set.values.reshape(-1, 1))
    val_set = scaler.fit_transform(val_set.values.reshape(-1, 1))
    return train_set, val_set

# Create Sequences
# Create for each row a sub list with 9 elemets for predict the 10
def subList(data, list_size:int = 9):
    X, y = [], []
    for i in range(len(data) - (list_size+1)):
        X.append(data[i:i+list_size])
        y.append(data[i+list_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

def euclidean_distance_loss(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.sum(torch.square(y_pred - y_true), axis=-1)))