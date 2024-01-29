import math
import torch
import numpy as np
import utils as utls
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self, pathTrain:str, pathval:str) -> None:
        ## TRAIN IMPORT DATASET
        self.data_train = utls.importDataset(pathTrain)
        ## val IMPORT DATASET
        self.data_val = utls.importDataset(pathval)
        # CONVERT IN SUBLIST TRAIN SET
        self.x_train, self.y_train = utls.subList(self.data_train)
        # CONVERT IN SUBLIST val SET
        self.x_val, self.y_val = utls.subList(self.data_train)


    def convertToTensor(self):
        ## CONVERT TO TENSOR TRAIN SET
        self.x_train = torch.tensor(self.x_train, dtype=torch.float64)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float64)
        # CONVERT TO TENSOR
        self.x_val = torch.tensor(self.x_val, dtype=torch.float64)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float64)
    
    def moveToGpu(self):
        # MOVE TENSOR TRAIN TO GPU
        self.x_train = self.x_train.to("cuda:0")
        self.y_train = self.y_train.to("cuda:0")
        # MOVE TENSOR TO GPU
        self.x_val = self.x_val.to("cuda:0")
        self.y_val = self.y_val.to("cuda:0")
        
    def createDataLoader(self, batch_train:int = 64, batch_val:int = 64) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        print("Batch size for training: ", batch_train)
        dataset_train = TensorDataset(self.x_train, self.y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_train, shuffle=True)
        # CREATE DATALOADER val
        print("Batch size for valing: ", batch_val)
        dataset_val = TensorDataset(self.x_val, self.y_val)
        data_loader_val = DataLoader(dataset_val, batch_size=batch_val, shuffle=True)
        return data_loader_train, data_loader_val
