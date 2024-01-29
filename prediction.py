import torch
import datetime
import numpy as np
import utils as utls
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set symbol
symbol = "BZ=F"
#Take current day
today = datetime.date.today()
#today = "2024-01-26"
start = "1980-01-01"
print(f"Downloading data for current day [{today}] to [{start}]")
# Download data
data = yf.download(symbol, 
                   start=start, 
                   end=today,
                   threads = True
)

# Preprocess data
data = data["Close"].dropna() # Remove all columns except "Close"
# Save data in csv
data.to_csv(f'datasets/{symbol}_prediction.csv', index = True)

scaler = MinMaxScaler()
data = scaler.fit_transform(data.values.reshape(-1, 1))

last_9_rows = data[-9:]
last_9_rows = np.reshape(last_9_rows, (1, -1))

# Load model
model = torch.load('models/testNet32-16-5000-FullData/model.pth')
model = model.to("cuda:0")
model.eval()
#
X = torch.tensor(last_9_rows)
X = X.to("cuda:0")
with torch.no_grad():
    prediction = model(X.unsqueeze(-1))
print(f"PRICE: {prediction}")
prediction_cpu = prediction.cpu().numpy()
prediction = scaler.inverse_transform(prediction_cpu)
print(f"PRICE: {prediction}")
