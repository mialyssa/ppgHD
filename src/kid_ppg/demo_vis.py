from demo import demo_utils
import matplotlib.pyplot as plt
import scipy
import numpy as np

# Data preparation
# Shows the demo data (Stair Exercise from S6 - PPGDalia)
# Visualization of the PPG acceleration across x, y, z
# axes in time freq domain

X, y = demo_utils.load_demo_data()

Y = scipy.fft.fft(X, axis = -1)
Y = np.abs(Y)[..., :128]

t = np.arange(Y.shape[0]) / 2
xf = scipy.fft.fftfreq(256, 1/32)[:128] * 60

# plt.figure()

# plt.subplot(2, 2, 1)
# plt.pcolormesh(t, xf[:60], Y[:, 0, :60].T)
# plt.ylabel('Freq. (BPM)')
# plt.title('PPG - Input Channel 1')
# plt.xticks([])

# plt.subplot(2, 2, 2)
# plt.pcolormesh(t, xf[1:60], Y[:, 1, 1:60].T)
# plt.title('ACCx - Input Channel 2')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(2, 2, 3)
# plt.pcolormesh(t, xf[1:60], Y[:, 2, 1:60].T)
# plt.ylabel('Freq. (BPM)')
# plt.title('ACCy - Input Channel 3')
# plt.xlabel('Time (sec)')

# plt.subplot(2, 2, 4)
# plt.pcolormesh(t, xf[1:60], Y[:, 3, 1:60].T)
# plt.title('ACCz - Input Channel 4')
# plt.yticks([])

#plt.show()
print("Finished Data preparation \n")


# Adaptive Filtering

# Removing Motion Artifact interference using
# acceleration signals as reference (AdaptiveFilteringModel)
# For linearly decoupling Blood Volume Pressure

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from preprocessing import sample_wise_z_score_normalization, sample_wise_z_score_denormalization
from adaptive_linear_model import AdaptiveFilteringModel

n_epochs = 30

cur_activity_X, ms, stds = sample_wise_z_score_normalization(X.copy())

sgd = tf.keras.optimizers.legacy.SGD(learning_rate = 1e-7,
                                            momentum = 1e-2,)
model = AdaptiveFilteringModel(local_optimizer = sgd,
                                num_epochs_self_train = n_epochs)

# Add visual bar here
X_filtered = model(cur_activity_X[..., None]).numpy()

X_filtered = X_filtered[:, None, :]
X_filtered = sample_wise_z_score_denormalization(X_filtered, ms, stds)

X_filtered = X_filtered[:, 0, :]


# Visualizing filtered PPG vs Original

Y_filtered = scipy.fft.fft(X_filtered, axis = -1)
Y_filtered = np.abs(Y_filtered)[..., :128]

t = np.arange(Y.shape[0]) / 2
xf = scipy.fft.fftfreq(256, 1/32)[:128] * 60

plt.figure()

plt.subplot(1, 2, 1)
plt.pcolormesh(t, xf[:60], Y[:, 0, :60].T)
plt.ylabel('Freq. (BPM)')
plt.title('Original PPG')
plt.xticks([])

plt.subplot(1, 2, 2)
plt.pcolormesh(t, xf[:60], Y_filtered[:, :60].T)
plt.title('PPG Filtered')
plt.xticks([])
plt.yticks([])

plt.suptitle('PPGDalia - S6 - Stairs Activity')

#plt.show()
print("Finished Adaptive Filtering \n")



# Probabilistic Heart Rate Extraction

# Using filtered (preprocessed) version of PPG,
# Run KID-PPG model to extract heart rate
# HR Distribution = soft estimates
# Applications: Y/N to retain sample => Error classifier
# predict() returns: expected HR, STD, and error classifier probability
# Chosen Threshold = 10 BPM

print("line 117 \n")

from kid_ppg import KID_PPG
from preprocessing import create_temporal_pairs
from hdcppg import KID_PPG_HDC #new 

#troublesome
X_filtered_temp, y_temp = create_temporal_pairs(X_filtered, y)
kid_ppg_model = KID_PPG()

hr_pred_m, hr_pred_std, hr_pred_p = kid_ppg_model.predict_threshold(X_filtered_temp, threshold = 10)




input_shape = X_filtered.shape 
ppg_model = KID_PPG_HDC(input_shape=input_shape, device='cpu') # X_filtered & y 

import torch
# Train the model
ppg_model.train(X_filtered, y)

# Test the model using the same data
predictions = []
actuals = []

for i in range(X_filtered.shape[0]):
    with torch.no_grad(): 
        x_seq = torch.Tensor(X_filtered[i, :]).to(ppg_model.hdc_model.device)
        y_true = torch.Tensor(y[i]).to(ppg_model.hdc_model.device)
        
        y_pred, _ = ppg_model.hdc_model._process_one_batch(X_filtered, y, i, mode="test")
        
        predictions.append(y_pred.item())  
        actuals.append(y_true.item())      

#convert lists to ndarray
predictions = np.array(predictions)
actuals = np.array(actuals)

print(f"Predictions: {predictions}")
print(f"Actuals: {actuals}")

print("Finished Heart Rate Extraction \n")


# Plotting results:
# Ground truth heart rate, 
# estimated expected heart rate, 
# and one standard deviation
t = np.arange(hr_pred_m.size) / 2

plt.figure()
plt.plot(t, y_temp, linewidth = 2, label = 'Ground Truth',
         color = 'C0')

plt.plot(t, hr_pred_m, linewidth = 2, label = 'KID-PPG',
         color = 'C1')
plt.fill_between(t, hr_pred_m - hr_pred_std,
                 hr_pred_m + hr_pred_std, alpha = 0.25,
                 color = 'C1')
plt.legend()

plt.xlabel('Time (sec.)')
plt.ylabel('Heart Rate (Beats Per Minute)')

plt.title('Pr')

# plt.show()


# Plotting samples which were retained with a 10 BPM threshold
plt.figure()
plt.plot(t, y_temp, linewidth = 2, label = 'Ground Truth',
         color = 'C0')

plt.plot(t, hr_pred_m, linewidth = 2, label = 'KID-PPG',
         color = 'C1')
plt.fill_between(t, hr_pred_m - hr_pred_std,
                 hr_pred_m + hr_pred_std, alpha = 0.25,
                 color = 'C1')
plt.plot(t[hr_pred_p > 0.5], hr_pred_m[hr_pred_p > 0.5], 'o',
         color = 'C2', label = 'KID-PPG (Thr = 10BPM)')

plt.legend()

plt.xlabel('Time (sec.)')
plt.ylabel('Heart Rate (Beats Per Minute)')
plt.title('Heart rate inference with retention')

# plt.show()
print("Finished Plotting Heart Rate Extraction \n")

print("FINISH ALL \n")
