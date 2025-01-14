import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Optional, Tuple
import scipy
import torchhd
from torchhd import embeddings

class HDCRegressor:
    def __init__(self, input_shape, device='cpu', learning_rate=1e-5, hvs_len = 10000):
        self.input_shape = input_shape
        self.device = torch.device(device)
     
        self.lr = learning_rate
        self.hvs_len = hvs_len
       
        self.dimensions = hvs_len
        self.lr = learning_rate
        self.M = torch.zeros(1, self.dimensions).to(self.device)
        # Projection embedding to encode input into hypervectors
        self.project = embeddings.Projection(input_shape[1], self.dimensions).to(self.device)
        
    def encode(self, x):
        return self.project(x)
        return torchhd.hard_quantize(res)
        
    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0,keepdim=True)
        self.M = update

    def forward(self, x) -> torch.Tensor:
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res  #prediction scalar 

    def train(self, X_filtered_temp, y, epochs=100):
    
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")  # Optional: Print the current epoch
            for i in tqdm(range(X_filtered_temp.shape[0])):  # Loop over each sample
                self._process_one_batch(X_filtered_temp, y, i, mode="train")
        
        
    def test(self, X_filtered: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        actuals = []

        for i in range(X_filtered.shape[0]):
            y_pred, y_true = self._process_one_batch(X_filtered, y, i, mode="test")
            predictions.append(y_pred.item())  
            actuals.append(y_true.item())      
        # Convert lists to ndarray
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        return predictions, actuals
    
    
    def _process_one_batch(
        self, data: np.ndarray, labels: np.ndarray, idx: int, mode: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:

        # inject noise
        # def inject(x_seq):
        #     return torch.normal(mean=x_seq, std=torch.full(x_seq.shape, 0.5))
        x_seq = torch.Tensor(data[idx, :]).to(self.device)  #256
        y_true = torch.Tensor(labels[idx]).to(self.device)  # Labels corresponding to the sequence
        
        # if mode == "test":
        #     x_seq = inject(x_seq)
     
        if mode == "train":
            encoded_x = self.encode(x_seq)
            self.model_update(encoded_x, y_true)

        if mode == "test":
            prediction = self.forward(x_seq) #no model update 
            return prediction, y_true
        

class KID_PPG_HDC:
    """KID-PPG class with HDC integration. Defines the KID-PPG model using HDC."""
    def __init__(self, input_shape, load_weights: bool = False, device='cpu', hvs_len=10000):
        """Initializes KID-PPG with HDC.

        Args:
            input_shape (tuple): Shape of the input data given to the KID-PPG model.
              Defaults to (N_samples, 256).
            load_weights (bool): True if pretrained weights should be loaded.
              Defaults to False.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            hvs_len (int): Length of the hyperdimensional space.
        """
        self.input_shape = input_shape
        self.hvs_len = hvs_len
        self.hdc_model = HDCRegressor(input_shape=input_shape, device=device, hvs_len=hvs_len)

        if load_weights:
            self.hdc_model.load_weights()  # Implement this if you have pre-trained weights

    def train(self, x: np.ndarray, y):
        """Trains the model using HDC.

        Args:
            x (numpy.ndarray): PPG signal for heart rate extraction.
              Size should be [N_samples, 256].
            y (numpy.ndarray): Corresponding labels.
              Size should be [N_samples, 1].
        """
        self.hdc_model.train(x, y)


    def test(self, x: np.ndarray, y):
        """Calls the test function in HDCRegressor to get predictions and actuals."""
        predictions, actuals = self.hdc_model.test(x,y)
        return predictions, actuals
        
