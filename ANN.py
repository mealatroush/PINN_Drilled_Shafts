import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
file_path = 'E:/Research/Research 2022/Research Projects/AI Geotech Project/Papers/Pile AXial + PINN/PINN Code/Cleaned_Dataset.xlsx'
data = pd.read_excel(file_path)

# Assuming the 'Case No' is the first column and 'Pult' is the last, we will adjust the indices accordingly
# Normalize the data except for the first ('Case No') and last ('Pult') columns
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = data.iloc[:, 1:-1].values  # Exclude 'Case No' and 'Pult'
y = data.iloc[:, -1].values.reshape(-1, 1)  # 'Pult'

# Fit the scaler on the features and target
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Define the model architecture
inputs = Input(shape=(X_scaled.shape[1],))
hidden1 = Dense(20, activation='ELU')(inputs)
hidden2 = Dense(20, activation='ELU')(hidden1)
hidden3 = Dense(10, activation='ELU')(hidden2)
outputs = Dense(1, activation='linear')(hidden3)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with the Nadam optimizer and standard MSE loss
model.compile(optimizer=Nadam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=1000, verbose=1)

# Prediction and plotting results
y_train_pred_scaled = model.predict(X_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_actual = scaler_y.inverse_transform(y_train)

# Plotting the actual vs predicted axial capacity
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label='Actual')
plt.plot(y_train_pred, label='Predicted')
plt.legend()
plt.title('Training Results')
plt.xlabel('Data Points')
plt.ylabel('Axial Capacity (Pult)')
plt.show()

# Plotting the loss history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plotting error (difference between actual and predicted)
error = y_actual - y_train_pred
plt.figure(figsize=(10, 5))
plt.plot(error, label='Error')
plt.legend()
plt.title('Prediction Error')
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.show()

# Calculate predictions for training and testing sets
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Inverse transform the predictions
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Calculate MSE and R2 for the training set
mse_train = mean_squared_error(scaler_y.inverse_transform(y_train), y_train_pred)
r2_train = r2_score(scaler_y.inverse_transform(y_train), y_train_pred)

# Calculate MSE and R2 for the test set
mse_test = mean_squared_error(scaler_y.inverse_transform(y_test), y_test_pred)
r2_test = r2_score(scaler_y.inverse_transform(y_test), y_test_pred)

# Print the metrics
print("Training MSE:", mse_train)
print("Training R2:", r2_train)
print("Testing MSE:", mse_test)
print("Testing R2:", r2_test)
