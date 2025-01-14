
import torch
from torch import nn
from tqdm import tqdm
from torchhd.models import Centroid
from torchhd import embeddings
import numpy as np
from torch import optim
from torch.nn import functional as F
from typing import Optional, Tuple
import torchhd
import torchmetrics

class KID_PPG_Centroid:
    def __init__(self, input_shape, device='cpu', hvs_len=10000, num_classes=None):
    
        self.input_shape = input_shape
        self.hvs_len = hvs_len
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.encoder = embeddings.Sinusoid(input_shape[1], hvs_len).to(self.device)
        self.model = Centroid(hvs_len, num_classes).to(self.device)

    def encode(self, x):
        return torchhd.hard_quantize(self.encoder(x))

    def train(self, X_filtered_train, y_train, epochs):
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in tqdm(range(X_filtered_train.shape[0])):  
                x_seq = torch.Tensor(X_filtered_train[i, :]).to(self.device)  # Input sequence
                y_true = torch.Tensor(y_train[i]).to(self.device).long()  # True labels
                y_true_rounded = torch.round(y_true).long()  
                encoded_x = self.encode(x_seq)
                self.model.add(encoded_x, y_true_rounded, lr = 0.00000005)

    def test(self, X_filtered_test, y_test):
        self.model.eval()
        predictions = []
        actuals = []
        #accuracy = torchmetrics.Accuracy("multiclass",num_classes=len(torch.unique(torch.tensor(y_test))))

        with torch.no_grad():
            #self.model.normalize()  
            for i in tqdm(range(X_filtered_test.shape[0])): 
                x_seq = torch.Tensor(X_filtered_test[i, :]).to(self.device) 
                y_true = torch.Tensor(y_test[i]).to(self.device).long() 

                encoded_x = self.encode(x_seq)
                outputs = self.model(encoded_x, dot=True)

                # Update accuracy metric
                #accuracy.update(outputs.cpu(), y_true)
    
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_true.cpu().numpy())

        #print(f"Testing accuracy: {accuracy.compute().item() * 100:.3f}%")
        return predictions, actuals