import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set symbol
symbol = "BZ=F"
# Download data
data = yf.download(symbol, 
                   start="1980-01-01", 
                   end="2024-01-05", 
                   threads = True
)

# Preprocess data
data = data["Close"].dropna() # Remove all columns except "Close"

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_set = data[:train_size]
test_set = data[train_size:]

# Save data in csv
train_set.to_csv(f'datasets/{symbol}_train.csv', index = True)
test_set.to_csv(f'datasets/{symbol}_test.csv', index = True)

# Normalize the data
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set.values.reshape(-1, 1))
test_set = scaler.fit_transform(test_set.values.reshape(-1, 1))

# Convert numpy array to dataframe
df_train = pd.DataFrame(train_set)
df_test = pd.DataFrame(test_set)

# Save data normalized in csv
df_train.to_csv(f'datasets/{symbol}_train_norm.csv', index = True)
df_test.to_csv(f'datasets/{symbol}_test_norm.csv', index = True)