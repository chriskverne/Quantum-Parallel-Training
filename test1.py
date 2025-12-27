import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# --- SETUP ---
dev = qml.device("default.qubit", wires=2)

def Operation_A(x):
    qml.RX(x, wires=1)

def Operation_B(x):
    qml.RZ(x * 3.0, wires=1)

@qml.qnode(dev)
def quantum_switch_circuit(x, theta):
    # 1. Prepare Control
    qml.RY(theta, wires=0)
    
    # 2. Switch Mechanism
    # Path 0: A -> B
    qml.ctrl(Operation_A, control=0, control_values=[0])(x)
    qml.ctrl(Operation_B, control=0, control_values=[0])(x)
    
    # Path 1: B -> A
    qml.ctrl(Operation_B, control=0, control_values=[1])(x)
    qml.ctrl(Operation_A, control=0, control_values=[1])(x)
    
    # 3. Interference
    qml.Hadamard(wires=0)

    # --- THE FIX IS HERE ---
    # We add a Hadamard to Wire 1 to rotate the basis. 
    # This makes the effect of Operation B (RZ) visible.
    qml.Hadamard(wires=1) 
    
    return qml.expval(qml.PauliZ(1))

# --- TRAINING ---
def cost(theta, X_batch, Y_batch):
    predictions = [quantum_switch_circuit(x, theta) for x in X_batch]
    # Standard MSE
    mse = np.mean((np.array(predictions) - Y_batch) ** 2)
    return mse

# Data Generation
X_data = np.linspace(-np.pi, np.pi, 50)
Y_data = np.sin(X_data) 

# Initialize Parameter
theta = np.array(0.5, requires_grad=True) 
opt = qml.AdamOptimizer(stepsize=0.1)

print("Training with Basis Rotation Fix...")
for i in range(101):
    theta, current_cost = opt.step_and_cost(lambda t: cost(t, X_data, Y_data), theta)
    if i % 5 == 0:
        print(f"Step {i}: Cost = {current_cost:.4f}, Theta = {theta.item():.4f}")

# --- PLOTTING ---
predictions = [quantum_switch_circuit(x, theta) for x in X_data]

plt.figure(figsize=(10,6))
plt.scatter(X_data, Y_data, label="Target", color="gray", alpha=0.5)
plt.plot(X_data, predictions, label="Quantum Switch", color="blue", linewidth=2)
plt.title(f"Fixed: Regression via Indefinite Causal Order\nFinal Theta: {theta.item():.2f}")
plt.legend()
plt.show()