import os
import torch
import NetMarket
import LoadData
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display


##HYPERPARAMS
learning_rates = 0.001
penality = 0.001
epochs = 1000

# PATH
pathTrain = "datasets/BZ=F_train_norm.csv"
pathTest = "datasets/BZ=F_test_norm.csv"

# IMPORT DATA
data = LoadData.Data(pathTrain, pathTest)
# DATA: TENSOR, GPU, DATALOADER
data.convertToTensor()
data.moveToGpu()
data_loader_train, data_loader_test = data.createDataLoader()

# Plot result
pathname = "models/test"
os.makedirs(pathname, exist_ok=True)
results = []
list_loss_train = []
list_loss_test = []

# Net model
model = NetMarket.LSTMNet()
# Move model to GPU
model = model.to("cuda:0")
# Model to float type
model = model.float()

# Settings loss fuction and optmizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=penality)

# TRAINING LOOP
model.train()  # Setting train evaluation

for epoch in range(epochs):
    # TRAIN METRICS
    total_loss = 0
    total_correct = 0
    total_len_train = 0

    ##TRAIN
    model.train()
    for x, y_true in data_loader_train:
        # Forward pass
        predictions = model(x)
        loss = loss_function(predictions, y_true.long())
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len_train += x.size(0)
    avg_loss_train = total_loss / len(data_loader_train)
    ## END TRAIN

    # TEST METRICS
    total_loss = 0
    total_correct = 0
    total_len_test = 0

    ## TEST
    model.eval()
    for x, y_true in data_loader_test:
        # Forward pass
        predictions = model(x)
        loss = loss_function(predictions, y_true.long())

        total_loss += loss.item()
        total_len_test += x.size(0)
    avg_loss_test = total_loss / len(data_loader_test)
    ##END TEST
    model.train()
    # PRINT RESULT
    result = f"Epoch [{epoch+1}/{epochs}], Loss-Train: {avg_loss_train:.4f}, Loss-Test: {avg_loss_test:.4f}"
    print(result)
    
    #add to list
    list_loss_train.append(avg_loss_train)
    list_loss_test.append(avg_loss_test)
    results.append(result)
## END EPOCH
    
#SAVE MODEL
torch.save(model, f'{pathname}/model.pth')
    
# Save plot loss
display.clear_output(wait=True)
plt.plot(list_loss_train, label='Training Loss')
plt.plot(list_loss_test, label = 'Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Epoch')
plt.legend()
plt.savefig(f'{pathname}/Loss.png')
plt.clf()
    
with open(f"{pathname}/results.txt", "w") as f:
    for result in results:
        f.write(result + "\n")