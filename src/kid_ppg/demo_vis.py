from demo import demo_utils
import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch


# load the data 
Xtrain, ytrain, _ = demo_utils.load_demo_data([1.0,2.0,3.0,4.0,5.0,6.0], split= "train")
Xtest, ytest, _ = demo_utils.load_demo_data([1.0,2.0,3.0,4.0,5.0,6.0], split= "test")
Xtrain_filtered = Xtrain[:, 0, :]
print("Xtrain_filtered shape: ", Xtrain_filtered.shape, "\n")
Xtest_filtered = Xtest[:, 0, :]
print("Xtest_filtered shape: ", Xtest_filtered.shape, "\n")

print("Finished Data preparation \n")



# data preprocessing 
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

# import tensorflow as tf
# from preprocessing import sample_wise_z_score_normalization, sample_wise_z_score_denormalization
# from adaptive_linear_model import AdaptiveFilteringModel

# n_epochs = 30

# cur_activity_X, ms, stds = sample_wise_z_score_normalization(X.copy())

# sgd = tf.keras.optimizers.SGD(learning_rate = 1e-7,
#                                             momentum = 1e-2,)
# model = AdaptiveFilteringModel(local_optimizer = sgd,
#                                 num_epochs_self_train = n_epochs)




# Add visual bar here
# X_filtered = model(cur_activity_X[..., None]).numpy()
# X_filtered = X_filtered[:, None, :]
# X_filtered = sample_wise_z_score_denormalization(X_filtered, ms, stds)
# X_filtered = X_filtered[:, 0, :]


# Visualizing filtered PPG vs Original
# Y_filtered = scipy.fft.fft(X_filtered, axis = -1)
# Y_filtered = np.abs(Y_filtered)[..., :128]

# t = np.arange(Y.shape[0]) / 2
# xf = scipy.fft.fftfreq(256, 1/32)[:128] * 60


# Round the heart rate values to integers
y_train_rounded = np.round(ytrain).astype(int)
y_test_rounded = np.round(ytest).astype(int)

# Calculate the minimum and maximum heart rate values across both sets
min_hr = np.min([np.min(y_train_rounded), np.min(y_test_rounded)])
max_hr = np.max([np.max(y_train_rounded), np.max(y_test_rounded)])

# Shift the heart rate values so that they start from 0
y_train_rounded -= min_hr
y_test_rounded -= min_hr

# Calculate the number of unique classes
num_classes = max_hr - min_hr + 1

from centroidppg import KID_PPG_Centroid
kid_ppg_model = KID_PPG_Centroid(input_shape=Xtrain_filtered.shape, device='cpu', hvs_len=10000, num_classes=num_classes)

# Train the model
kid_ppg_model.train(Xtrain_filtered, y_train_rounded, epochs=50)
# Test the model
predictions, actuals = kid_ppg_model.test(Xtest_filtered, y_test_rounded)



# Remap the prediction classes back to normal heart rate:
actuals = ytest
predictions = np.squeeze(np.array(predictions))
predicted_classes = np.argmax(predictions, axis=-1)
print(f"Predicted classes shape: {predicted_classes.shape}\n")

predictions_mapped = predicted_classes + min_hr 
print("predictions_mapped shape: ", predictions_mapped.shape, "\n")  
print("first 10 predictions: ", predictions_mapped[:10], "\n") 



# # Initialize model 
# from hdcppg import KID_PPG_HDC 
# input_shape = X_filtered.shape 
# ppg_model = KID_PPG_HDC(input_shape=input_shape, device='cpu') # X_filtered & y 

# # Train the model
# ppg_model.train(X_filtered, y)





# predictions = []
# actuals = []
# # Test the model 
# #predictions, actuals = ppg_model.test(X_filtered, y)

# for i in range(X_filtered.shape[0]):
#     with torch.no_grad(): 
#         x_seq = torch.Tensor(X_filtered[i, :]).to(ppg_model.hdc_model.device)
#         y_true = torch.Tensor(y[i]).to(ppg_model.hdc_model.device)
#         y_pred, _ = ppg_model.hdc_model._process_one_batch(X_filtered, y, i, mode="test")
#         predictions.append(y_pred.item())  
#         actuals.append(y_true.item())      

# #convert lists to ndarray
# predictions = np.array(predictions)
# actuals = np.array(actuals)
# print(f"Predictions: {predictions}")
# print(f"Actuals: {actuals}")

# print("Finished Heart Rate Extraction \n")


# Plotting results:
# Ground truth heart rate, 
# estimated expected heart rate, 
# and one standard deviation
t = np.arange(actuals.size) / 2

plt.figure()
plt.plot(t, actuals, linewidth = 2, label = 'Ground Truth', color = 'C0')

plt.plot(t, predictions_mapped, linewidth = 2, label = 'KID-PPG-HD', color = 'C1')
# plt.fill_between(t, hr_pred_m - hr_pred_std, hr_pred_m + hr_pred_std, alpha = 0.25, color = 'C1')
plt.legend()

plt.xlabel('Time (sec.)')
plt.ylabel('Heart Rate (Beats Per Minute)')
plt.title('Pr')
plt.show()
print("Finished Plotting Heart Rate Extraction \n")

print("FINISH ALL \n")