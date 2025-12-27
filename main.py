# import pennylane as qml
# from pennylane import numpy as np
# import threading
# import time
# import random

# # --- 1. SETUP: The Physics (Ising Model) ---
# n_qubits = 4
# # Simple Transverse Field Ising Model Hamiltonian
# coeffs = [-1.0] * n_qubits + [-0.5] * n_qubits
# obs = [qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits) for i in range(n_qubits)]
# obs += [qml.PauliX(i) for i in range(n_qubits)]
# H = qml.Hamiltonian(coeffs, obs)

# # We use 'default.qubit' (noiseless) as requested.
# # In a real scenario, this would be a QPU connection.
# dev = qml.device("default.qubit", wires=n_qubits)

# # --- 2. THE ANSATZ (Model) ---
# # Simple Rx, Rz, Entangling layers
# def circuit(params, wires):
#     qml.StronglyEntanglingLayers(weights=params, wires=wires)

# @qml.qnode(dev)
# def cost_fn(params):
#     circuit(params, wires=range(n_qubits))
#     return qml.expval(H)

# # --- 3. SHARED GLOBAL STATE ---
# # We need a shared parameter array.
# # Shape: (layers, n_qubits, 3 parameters per qubit for StrongEnt)
# n_layers = 1
# shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
# global_params = np.random.random(shape, requires_grad=True)

# # A simplified lock just to ensure array write safety (Python GIL handles most, but this is good practice)
# param_lock = threading.Lock()

# # Hyperparameters
# learning_rate = 0.1
# max_steps = 15  # Low number for the demo
# print(f"Initial Cost: {cost_fn(global_params):.4f}")

# # --- 4. THE ASYNCHRONOUS WORKER ---
# def async_worker(worker_id, assigned_indices):
#     """
#     Simulates a QPU that calculates gradients for a specific subset of parameters.
#     See 'Strategy 1: Coordinate Asynchrony'.
#     """
#     global global_params
    
#     for step in range(max_steps):
#         # A. Pull Global Parameters
#         # We assume we work on the current snapshot of parameters
#         with param_lock:
#             current_params = np.array(global_params, requires_grad=True)

#         # B. Simulate QPU Latency (The 'Straggler' effect)
#         # Some QPUs are slower than others due to queueing/calibration 
#         delay = random.uniform(0.01, 0.1) 
#         time.sleep(delay) 
        
#         # C. Compute Gradient for ASSIGNED parameters only
#         # We manually implement a partial parameter shift for efficiency here
#         # or just let PennyLane compute the full grad and mask it (simpler for code).
#         grad_fn = qml.grad(cost_fn)
#         full_grad = grad_fn(current_params)
        
#         # Mask: We only apply the update for the parameters this worker owns.
#         # This creates the 'Parameter Parallelism'[cite: 21].
        
#         # D. Push Update (Asynchronous)
#         with param_lock:
#             # We apply the gradient to the GLOBAL parameters, which might have changed 
#             # while we were sleeping (Staleness!).
            
#             # Update only assigned indices
#             # (Flattening for easier indexing in this demo)
#             flat_params = global_params.flatten()
#             flat_grad = full_grad.flatten()
            
#             for idx in assigned_indices:
#                 flat_params[idx] -= learning_rate * flat_grad[idx]
            
#             # Reshape back
#             global_params = flat_params.reshape(shape)
            
#             # Optional: Print progress
#             print(f"Worker {worker_id} updated indices {assigned_indices}. "
#                   f"Cost: {cost_fn(global_params):.4f}")

# # --- 5. ORCHESTRATION ---

# # Distribute parameters among 3 'QPUs'
# total_params = np.prod(shape)
# indices = list(range(total_params))
# # Split indices into 3 chunks
# chunks = [indices[i::3] for i in range(3)] 

# threads = []
# print(f"\nStarting training with 3 Asynchronous 'QPUs' on {total_params} parameters...\n")

# for i in range(3):
#     t = threading.Thread(target=async_worker, args=(i, chunks[i]))
#     threads.append(t)
#     t.start()

# # Wait for all to finish
# for t in threads:
#     t.join()

# print("\nTraining Complete.")
# print(f"Final Cost: {cost_fn(global_params):.4f}")





import pennylane as qml
from pennylane import numpy as np

# --- 1. Setup ---
# We want to rotate |0> to |1> (The "Target")
# We will use a simple Ry(theta) gate.
# Standard Gradient Descent would calculate d(Loss)/d(theta).
# WE will use "Back-Projection": Invert the circuit and measure the gap at the input.

dev = qml.device("default.qubit", wires=1)

# The Target State we want to reach: |1>
# On the Bloch sphere, |0> is at (0, 0, 1), |1> is at (0, 0, -1). Angle diff is pi.
TARGET_STATE = np.array([0, 1]) 

@qml.qnode(dev)
def forward_circuit(theta):
    """Run the circuit forward with current parameters."""
    qml.RY(theta, wires=0)
    return qml.state()

@qml.qnode(dev)
def inverse_circuit_on_target(theta):
    """
    The 'Back-Projection': 
    Initialize the qubit in the TARGET state (|1>), 
    then run the INVERSE circuit (Ry(-theta)).
    Returns: The 'Required Input' state.
    """
    # FIX: QubitStateVector is renamed to StatePrep in newer PennyLane versions
    qml.StatePrep(TARGET_STATE, wires=0)
    
    # Run Inverse Circuit (Adjoint)
    # Note: qml.adjoint(Op)(params) is the correct syntax
    qml.adjoint(qml.RY)(theta, wires=0)
    
    return qml.state()


def get_bloch_angle(state_vector):
    """
    Helper: Calculate the angle 'theta' of a state vector on the Bloch sphere 
    relative to |0>. 
    State = cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>
    """
    # Get amplitude of |1> (index 1)
    # We assume real coefficients for this simple Ry rotation demo
    c0 = state_vector[0].real
    c1 = state_vector[1].real
    
    # tan(theta/2) = c1/c0  -> theta = 2 * arctan2(c1, c0)
    return 2 * np.arctan2(c1, c0)

# --- 2. The Iterative Optimization Loop ---

# Start with a random bad parameter (e.g., 0.1 radians)
theta = np.array(0.1, requires_grad=False) 

print(f"Start Theta: {theta:.4f}")
print("-" * 30)

for step in range(5):
    # 1. Forward Pass (Just to see where we are)
    current_state = forward_circuit(theta)
    prob_1 = current_state[1]**2
    print(f"Step {step}: Output Prob(|1>) = {prob_1:.4f}")

    if prob_1 > 0.999:
        print("Converged!")
        break

    # 2. Back-Projection (The Core of Method B)
    # We ask: "Given our current gate Ry(theta), what input would have produced |1>?"
    required_input_state = inverse_circuit_on_target(theta)
    
    # 3. Geometric Update (The "Projection")
    # We measure the gap between the 'Required Input' and our 'Actual Input' (|0>)
    # Actual Input |0> corresponds to angle 0.
    required_input_angle = get_bloch_angle(required_input_state)
    
    # The Update Rule: Shift theta by exactly the gap amount.
    # New Theta = Old Theta + Gap
    print(f"   -> Back-Projected Gap at Input: {required_input_angle:.4f} rad")
    theta += required_input_angle

print("-" * 30)
print(f"Final Theta: {theta:.4f} (Target was pi = {np.pi:.4f})")