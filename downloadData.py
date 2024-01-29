import datetime
import utils as utls
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set symbol
symbol = "BZ=F"
#Take current day
today= datetime.date.today()
print(f"Downloading date for current day [{today}]")
# Download data
data = yf.download(symbol, 
                   start="1980-01-01", 
                   end=today,
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

# Normalize data
train_set, test_set = utls.normalizeData(train_set, test_set)

# Convert numpy array to dataframe
df_train = pd.DataFrame(train_set, columns = ['Close'],)
df_test = pd.DataFrame(test_set, columns = ['Close'])

# Save data normalized in csv
df_train.to_csv(f'datasets/{symbol}_train_norm.csv', index = True)
df_test.to_csv(f'datasets/{symbol}_test_norm.csv', index = True)