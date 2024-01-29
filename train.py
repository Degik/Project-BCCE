import os
import torch
import LoadData
import NetMarket
import statistics
import utils as utls
import torch.nn as nn
import matplotlib.pyplot as plt
import IPython.display as display


##HYPERPARAMS
learning_rate = 0.0001
penality = 0.0001
epochs = 5000

# PATH
pathTrain = "datasets/BZ=F_train_norm.csv"
pathval = "datasets/BZ=F_val_norm.csv"

# IMPORT DATA
data = LoadData.Data(pathTrain, pathval)
# DATA: TENSOR, GPU, DATALOADER
data.convertToTensor()
data.moveToGpu()
data_loader_train, data_loader_val = data.createDataLoader(batch_train=64, batch_val=64)

# Plot result
pathname = "models/testNet32-16-5000-FullData"
os.makedirs(pathname, exist_ok=True)
results = []
list_loss_train = []
list_loss_val = []
# Distance list
euclidean_distance_train = []
euclidean_distance_val = []

# Net model
model = NetMarket.LSTMModel()
# Move model to GPU
model = model.to("cuda:0")
# Model to double type
model = model.double()

# Settings loss fuction and optmizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=penality)

# TRAINING LOOP
model.train()  # Setting train evaluation

for epoch in range(epochs):
    # TRAIN METRICS
    total_loss = 0
    total_correct = 0
    total_len_train = 0
    #
    train_distance = []
    val_distance = []
    ##TRAIN
    model.train()
    for x, y_true in data_loader_train:
        # Forward pass
        predictions = model(x.unsqueeze(-1))
        loss = loss_function(predictions, y_true.unsqueeze(-1))
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len_train += x.size(0)

        distance = utls.euclidean_distance_loss(y_true.unsqueeze(-1), predictions)
        train_distance.append(distance.item())
    avg_loss_train = total_loss / len(data_loader_train)
    # distance mean
    mean_distance_train = statistics.mean(train_distance)
    # distance list
    euclidean_distance_train.append(mean_distance_train)

    ## END TRAIN

    # VAL METRICS
    total_loss = 0
    total_correct = 0
    total_len_val = 0

    ## VAL
    model.eval()
    with torch.no_grad():
        for x, y_true in data_loader_val:
            # Forward pass
            predictions = model(x.unsqueeze(-1))
            loss = loss_function(predictions, y_true.unsqueeze(-1))

            total_loss += loss.item()
            total_len_val += x.size(0)

            distance = utls.euclidean_distance_loss(y_true.unsqueeze(-1), predictions)
            val_distance.append(distance.item())
        avg_loss_val = total_loss / len(data_loader_val)

    # distance mean
    mean_distance_val = statistics.mean(val_distance)
    # distance list
    euclidean_distance_val.append(mean_distance_val)

    ##END VAL
    model.train()
    # PRINT RESULT
    result = f"Epoch [{epoch+1}/{epochs}], Loss-Train: {avg_loss_train:.4f}, Loss-val: {avg_loss_val:.4f}, MEE-Train: {mean_distance_train:.4f}, MEE-Val{mean_distance_val:.4f}"
    print(result)
    
    #add to list
    list_loss_train.append(avg_loss_train)
    list_loss_val.append(avg_loss_val)
    results.append(result)
## END EPOCH
    
#SAVE MODEL
torch.save(model, f'{pathname}/model.pth')
    
# Save plot loss
display.clear_output(wait=True)
plt.plot(list_loss_train, label='Training Loss')
plt.plot(list_loss_val, label = 'Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.004])
plt.title('Loss for Epoch')
plt.legend()
plt.savefig(f'{pathname}/Loss.png')
plt.clf()

# Save plot MEE
display.clear_output(wait=True)
plt.plot(euclidean_distance_train, label='MEE-Training')
plt.plot(euclidean_distance_val, label = 'MEE-Val')
plt.xlabel('Epoch')
plt.ylabel('MEE')
plt.ylim([0, 0.05])
plt.title('MEE for Epoch')
plt.legend()
plt.savefig(f'{pathname}/MEE.png')
plt.clf()
    
with open(f"{pathname}/results.txt", "w") as f:
    for result in results:
        f.write(result + "\n")