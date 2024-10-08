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
    def __init__(self, input_shape, device='cpu', l2_lambda=2e-3, learning_rate=1e-8, hvs_len = 10000 ):
        self.input_shape = input_shape
        self.device = torch.device(device)
        self.l2_lambda = l2_lambda
        self.lr = learning_rate
        self.hvs_len = hvs_len
        self.args = {
            'l2_lambda': l2_lambda
        }
        self.dimensions = hvs_len
        self.lr = learning_rate
        self.M = torch.zeros(1, self.dimensions).to(self.device)
        # Projection embedding to encode input into hypervectors
        self.project = embeddings.Projection(input_shape[1], self.dimensions).to(self.device)
        
    def encode(self, x):
        #Encode input sequence into a hv
        return self.project(x)
        #return torchhd.hard_quantize(sample_hv)
        
    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0,keepdim=True)
        self.M = update

    def forward(self, x) -> torch.Tensor:
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res

    def train(self, X_filtered_temp, y, epochs=100):
        train_data = X_filtered_temp
        label = y
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")  # Optional: Print the current epoch
            for i in tqdm(range(train_data.shape[0])):  # Loop over each sample
                self._process_one_batch(train_data, label, i, mode="train")
        print("training epochs:", epochs)
        print("l2 lambda: ", self.l2_lambda)
        print("learning rate: ", self.lr)
        print("hvs_len: ", self.hvs_len)
        
    def test(self, X_filtered: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        actuals = []

        for i in range(X_filtered.shape[0]):
            with torch.no_grad(): 
                x_seq = torch.Tensor(X_filtered[i, :]).to(self.device)  # shape (256,)
                y_true = torch.Tensor(y[i]).to(self.device)
                
               
                y_pred, _ = self._process_one_batch(X_filtered, y, i, mode="test")

                predictions.append(y_pred.item())  
                actuals.append(y_true.item())      

        # print(predictions)
        # print(actuals)
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
            encoded_hv = self.encode(x_seq)
            self.model_update(encoded_hv, y_true)

        if mode == "test":
            y_pred = self.forward(x_seq)
            return y_pred, y_true
        
        if mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose either 'train' or 'test'.")


class KID_PPG_HDC:
    """KID-PPG class with HDC integration. Defines the KID-PPG model using HDC."""
    def __init__(self, input_shape, load_weights: bool = False, device='cpu', hvs_len=30000):
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
        
    
    def predict_threshold(self, x: np.ndarray, threshold: np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimates HR probability given PPG input and calculates the probability of error.

        Args:
            x (numpy.ndarray): PPG signal for heart rate extraction.
              Size should be [N_samples, 256].
            threshold (numpy.float32): Threshold for estimating the probability
              p(error > threshold).
        Returns:
            y_pred_m (numpy.ndarray): Expected HR as estimated by KID-PPG model.
              Size is [N_samples, 1].
            y_pred_std (numpy.ndarray): Estimated standard deviation of the
              HR distribution. Size is [N_samples, 1].
            p_error (numpy.ndarray): Estimated probability of error > threshold.
              Size is [N_samples, 1].
        """
        y_pred_m, y_pred_std = self.predict(x)  # Use the train method instead if needed

        # Calculate the probability of error using the threshold
        p_error = scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m + threshold) \
                  - scipy.stats.norm(y_pred_m, y_pred_std).cdf(y_pred_m - threshold)

        return y_pred_m, y_pred_std, p_error
