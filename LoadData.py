import math
import torch
import numpy as np
import utils as utls
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self, pathTrain:str, pathTest:str) -> None:
        ## TRAIN IMPORT DATASET
        self.data_train = utls.importDataset(pathTrain)
        ## TEST IMPORT DATASET
        self.data_test = utls.importDataset(pathTest)
        # CONVERT IN SUBLIST TRAIN SET
        self.x_train, self.y_train = utls.subList(self.data_train)
        # CONVERT IN SUBLIST TEST SET
        self.x_test, self.y_test = utls.subList(self.data_train)


    def convertToTensor(self):
        ## CONVERT TO TENSOR TRAIN SET
        self.x_train = torch.tensor(self.x_train, dtype=torch.float64)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float64)
        # CONVERT TO TENSOR
        self.x_test = torch.tensor(self.x_test, dtype=torch.float64)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float64)
    
    def moveToGpu(self):
        # MOVE TENSOR TRAIN TO GPU
        self.x_train = self.x_train.to("cuda:0")
        self.y_train = self.y_train.to("cuda:0")
        # MOVE TENSOR TO GPU
        self.x_test = self.x_test.to("cuda:0")
        self.y_test = self.y_test.to("cuda:0")
        
    def createDataLoader(self, batch_train:int = 64, batch_test:int = 64) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        print("Batch size for training: ", batch_train)
        dataset_train = TensorDataset(self.x_train, self.y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_train, shuffle=True)
        # CREATE DATALOADER TEST
        print("Batch size for testing: ", batch_test)
        dataset_test = TensorDataset(self.x_test, self.y_test)
        data_loader_test = DataLoader(dataset_test, batch_size=batch_test, shuffle=True)
        return data_loader_train, data_loader_test
