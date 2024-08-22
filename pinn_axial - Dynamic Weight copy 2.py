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

# Normalize the data except for target column 'Pult'
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = data.iloc[:, 1:-1].values  # Exclude 'Case No' and 'Pult'
y = data.iloc[:, -1].values.reshape(-1, 1)  # 'Pult'

# Fit the scaler on the features and target
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Define the custom loss function incorporating the physical equation
def custom_loss(y_true, y_pred):
    # Unscale the features to calculate Pult using the physical equation
    unscaled_inputs = scaler_X.inverse_transform(X_train)

    # Extract parameters from the unscaled inputs
    D = unscaled_inputs[:, 0]
    L = unscaled_inputs[:, 1]
    C = unscaled_inputs[:, 2]
    delta = unscaled_inputs[:, 3]
    K0 = unscaled_inputs[:, 4]
    gamma = unscaled_inputs[:, 5]

    # Convert angles from degrees to radians for the calculations
    delta_rad = np.deg2rad(delta)

    # Bearing capacity factors calculations
    Nq = np.exp(np.pi * np.tan(delta_rad)) * np.tan(np.pi / 4 + delta_rad / 2)**2
    Nc = 1 / np.tan(delta_rad) * (Nq - 1)
    Nγ = (Nq - 1) * np.tan(1.4 * delta_rad)

    # Calculation of As and Ab based on D and L
    As = np.pi * D * L
    Ab = np.pi * (D / 2)**2

    # Calculate Pult using the provided physical equation
    Pult_calculated = As * (C + (K0 * gamma * L / 2 * np.tan(delta_rad))) + \
                      Ab * ((C * Nc) + (gamma * L * Nq) + (gamma * D / 2 * Nγ))

    # Scale Pult_calculated to match the scaled predictions
    Pult_calculated_scaled = scaler_y.transform(Pult_calculated.reshape(-1, 1))

    # Physics-informed loss (50% weight)
    physics_loss = 1 * tf.reduce_mean(tf.square(y_pred - Pult_calculated_scaled))

    # Data-driven loss (100% weight)
    data_loss = 1 * tf.reduce_mean(tf.square(y_true - y_pred))

    # Combined loss
    return data_loss + physics_loss

# Define the model architecture
inputs = Input(shape=(X_scaled.shape[1],))
hidden1 = Dense(20, activation='ELU')(inputs)
hidden2 = Dense(20, activation='ELU')(hidden1)
hidden3 = Dense(10, activation='ELU')(hidden2)
outputs = Dense(1, activation='linear')(hidden3)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with the Nadam optimizer and the custom loss function
optimizer = Nadam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_loss)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=1000, verbose=1)

# Prediction and plotting results
y_train_pred_scaled = model.predict(X_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

# Plotting the actual vs predicted axial capacity
plt.figure(figsize=(10, 5))
plt.plot(scaler_y.inverse_transform(y_train), label='Actual')
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

# Inverse transform the scaled test features and target values to their original scales
unscaled_X_test = scaler_X.inverse_transform(X_test)
y_test_actual = scaler_y.inverse_transform(y_test)  # Actual Pult values for the test set

# Original Bearing Capacity Factors Calculations (Repeating for clarity)
delta_rad = np.deg2rad(unscaled_X_test[:, 3])
Nq_original = np.exp(np.pi * np.tan(delta_rad)) * np.tan(np.pi / 4 + delta_rad / 2)**2
Nc_original = np.where(delta_rad != 0, 1 / np.tan(delta_rad) * (Nq_original - 1), 0)
Ngamma_original = (Nq_original - 1) * np.tan(1.4 * delta_rad)

# Assuming simplified formulas for As and Ab based on D and L for demonstration
D = unscaled_X_test[:, 0]
L = unscaled_X_test[:, 1]
C = unscaled_X_test[:, 2]
K0 = unscaled_X_test[:, 4]
gamma = unscaled_X_test[:, 5]

# Calculate Pult using the original bearing capacity factors
As = np.pi * D * L
Ab = np.pi * (D / 2)**2
Pult_calculated_with_original_factors = As * (C + (K0 * gamma * L / 2 * np.tan(delta_rad))) + \
                                        Ab * ((C * Nc_original) + (gamma * L * Nq_original) + (gamma * D / 2 * Ngamma_original))

# Direct comparison of Pult values: Model Predictions vs. Calculated using Original Equations
# Adjust the plotting to use line plots
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Pult', color='blue', linewidth=2)
plt.plot(y_test_pred, label='Model Predicted Pult', color='red', linestyle='--', linewidth=2)
plt.plot(Pult_calculated_with_original_factors, label='Calculated Pult Using Original Equations', color='green', linestyle=':', linewidth=2)
plt.title('Comparison of Pult: Actual vs. Predicted vs. Calculated Using Original Equations')
plt.xlabel('Sample Index')
plt.ylabel('Axial Capacity (Pult)')
plt.legend()
plt.show()




