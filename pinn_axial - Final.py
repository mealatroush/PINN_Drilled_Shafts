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
X = data.drop(columns=['Case No', 'Pult']).values  # Exclude 'Case No' and 'Pult'
y = data['Pult'].values.reshape(-1, 1)  # 'Pult'

# Fit the scaler on the features and target
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Define the custom loss function incorporating the physical equation
def custom_loss(y_true, y_pred):
    # Unscale the features to calculate Pult using the physical equation
    unscaled_inputs = scaler_X.inverse_transform(X_train)

    # Extract parameters from the unscaled inputs
    D = unscaled_inputs[:, 0]  # Diameter
    L = unscaled_inputs[:, 1]  # Length
    C = unscaled_inputs[:, 2]  # Cohesion
    delta = unscaled_inputs[:, 3]  # Friction Angle
    K0 = unscaled_inputs[:, 4]  # Earth Pressure Coefficient
    gamma = unscaled_inputs[:, 5]  # Unit Weight of Soil
    psi = unscaled_inputs[:, 6]  # Dilatancy Angle
    E = unscaled_inputs[:, 7]  # Young's Modulus

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

    # Physics-informed loss (100% weight)
    physics_loss = 1.0 * tf.reduce_mean(tf.square(y_pred - Pult_calculated_scaled))

    # Data-driven loss (100% weight)
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))

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
y_actual = scaler_y.inverse_transform(y_train)

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

# Define your input labels based on the dataset columns (excluding 'Case No' and 'Pult')
input_labels = ['D', 'L', 'C', 'delta', 'K0', 'gamma', 'ψ', 'E', 'R']

# Automatically generate a conceptual equation-like representation (simplified)
def generate_equation(model, input_labels):
    print("Conceptual equation-like representation:")
    for layer in model.layers[1:]:  # Skip the input layer
        weights, biases = layer.get_weights()
        for neuron_idx in range(weights.shape[1]):
            print(f"Layer {layer.name}, Neuron {neuron_idx}:")
            equation_parts = [f"({weights[i, neuron_idx]} * {label})" for i, label in enumerate(input_labels)]
            equation = " + ".join(equation_parts) + f" + {biases[neuron_idx]}"
            print(equation)
            print()  # Newline for readability

# Generate the conceptual equation-like representation
generate_equation(model, input_labels)

# Extract weights and biases and plot DataFrame as a table
# This section remains the same as in the previous version

# Assuming 'model' is your trained model

def extract_weights_biases(model):
    layers_data = []

    for layer in model.layers[1:]:  # Skip the Input layer
        weights, biases = layer.get_weights()
        for neuron_idx in range(weights.shape[1]):
            neuron_data = {'Layer': layer.name, 'Neuron': neuron_idx}
            for input_idx, weight in enumerate(weights[:, neuron_idx]):
                neuron_data[f'Weight_{input_idx}'] = weight
            neuron_data['Bias'] = biases[neuron_idx]
            layers_data.append(neuron_data)

    return pd.DataFrame(layers_data)

# Extract weights and biases
df_weights_biases = extract_weights_biases(model)

# Display the DataFrame
print(df_weights_biases)

# Optional: Save the DataFrame to a CSV file
df_weights_biases.to_csv("weights_biases.csv", index=False)

# Plotting a table (assuming df_weights_biases is not too large)
fig, ax = plt.subplots(figsize=(12, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_weights_biases.values, colLabels=df_weights_biases.columns, loc='center')

plt.show()