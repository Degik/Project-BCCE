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
trainSet = data[:train_size]
testSet = data[train_size:]

#Save the date in csv
trainSet.to_csv(f'datasets/{symbol}_train.csv', index = True)
testSet.to_csv(f'datasets/{symbol}_test.csv', index = True)

#Normalize the data
scaler  = MinMaxScaler()