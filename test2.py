import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Use a device that supports mid-circuit measurements
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def zeno_activation_circuit(x, params):
    theta_measure = params[0]  # The "Activation Angle"
    theta_correct = params[1]  # The "Weight" adjustment
    
    # 1. Encode Data (Standard)
    qml.RX(x, wires=0)
    
    # 2. Entangle with Ancilla (The "Synapse")
    qml.CNOT(wires=[0, 1])
    
    # 3. TRAINABLE MEASUREMENT (The "Activation")
    # We rotate the ancilla before measuring.
    # This changes the basis of the measurement collapse.
    qml.RY(theta_measure, wires=1)
    
    # Measure the ancilla (Mid-circuit)
    m_0 = qml.measure(1)
    
    # 4. CONDITIONAL OPERATION (Feed-forward)
    # If we measured 1, we 'kick' the data qubit.
    # This creates a non-linear "kink" in the function.
    qml.cond(m_0, qml.RX)(theta_correct, wires=0)
    
    # 5. Final Prediction
    return qml.expval(qml.PauliZ(0))

# --- TRAINING ---
def cost(params, X_batch, Y_batch):
    preds = np.array([zeno_activation_circuit(x, params) for x in X_batch])
    return np.mean((preds - Y_batch) ** 2)

# Data: A Sine Wave (Hard for linear gates, easy for non-linear)
X_data = np.linspace(-np.pi, np.pi, 50)
Y_data = np.sin(X_data) 

# Params: [Measurement Angle, Correction Angle]
# Initialize slightly off-center to avoid symmetry traps
params = np.array([0.5, 0.5], requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.1)

print("Training the Zeno Activation...")
for i in range(51):
    params, c = opt.step_and_cost(lambda p: cost(p, X_data, Y_data), params)
    if i % 10 == 0:
        print(f"Step {i}: Cost = {c:.4f} | Meas Angle={params[0]:.2f}, Correction={params[1]:.2f}")

# --- VISUALIZATION ---
final_preds = [zeno_activation_circuit(x, params) for x in X_data]

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data, "k--", label="Target (Sine)")
plt.plot(X_data, final_preds, "r-", linewidth=2, label="Zeno Prediction")
plt.title(f"Trainable Measurement Basis\nParams: {params.numpy().round(2)}")
plt.legend()
plt.show()