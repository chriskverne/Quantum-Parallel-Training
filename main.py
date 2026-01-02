# import pennylane as qml
# from pennylane import numpy as np

# # --- 1. SETUP: Simple VQE Circuit ---
# # We use a simple 2-qubit system trying to find the ground state of a Hamiltonian.
# # Hamiltonian: H = Z0 * Z1 (Ground state is |11> with energy -1, or |00> with energy 1 depending on signs)
# # Let's use H = 1.0 * Z(0) @ Z(1) + 0.5 * X(0)
# coeffs = [1.0, 0.5]
# obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
# H = qml.Hamiltonian(coeffs, obs)

# # Device
# dev = qml.device("default.qubit", wires=3)

# # Ansatz: Strong entanglement to ensure non-trivial geometry
# @qml.qnode(dev)
# def cost_fn(theta):
#     qml.StronglyEntanglingLayers(theta, wires=[0, 1])
#     return qml.expval(H)

# # Auxiliary qnode to get the state for metric tensor calculation (if needed manually)
# # But PennyLane has qml.metric_tensor built-in.

# # --- 2. THEORETICAL CONSTANTS ---
# # To compute the bound, we need the norms of the generators and observables.
# # For StronglyEntanglingLayers, generators are Pauli rotations (norm = 0.5)
# GENERATOR_NORM = 0.5 
# NUM_LAYERS = 1
# NUM_WIRES = 2
# # Shape of theta is (Layers, Wires, 3)
# SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=NUM_LAYERS, n_wires=NUM_WIRES)
# NUM_PARAMS = np.prod(SHAPE)

# # Bound on the Observable Norm ||M||_2
# # For Pauli sums, strictly ||M|| <= sum(|coeffs|). 
# M_NORM = np.sum(np.abs(coeffs)) 

# # L_Euc (Euclidean Smoothness Constant)
# # L <= 4 * ||M|| * sum(||G_k||^2)
# L_EUC = 4 * M_NORM * (NUM_PARAMS * (GENERATOR_NORM ** 2))

# # K (Riemannian Correction Constant)
# # Heuristic bound based on derivative of metric tensor (depends on G^3)
# K_CONST = 4 * NUM_PARAMS * M_NORM * (GENERATOR_NORM ** 3)

# # --- 3. ADAPTIVE REGULARIZED QNG OPTIMIZER ---

# def adaptive_qng_step(theta, current_step_idx, epsilon=0.01):
#     """
#     Performs one step of Regularized QNG with the theoretically safe step size.
#     """
#     # A. Calculate Gradient (Euclidean)
#     grad_fn = qml.grad(cost_fn)
#     grad = grad_fn(theta)
#     grad_norm = np.linalg.norm(grad)
    
#     # B. Calculate Metric Tensor (Riemannian)
#     # qml.metric_tensor returns the block-diagonal approximation by default
#     mt_fn = qml.metric_tensor(cost_fn)
#     g = mt_fn(theta)
    
#     # Reshape metric to square matrix (N x N)
#     g_matrix = np.reshape(g, (NUM_PARAMS, NUM_PARAMS))
    
#     # C. Calculate Eigenvalues for the bound
#     # We need lambda_max of the metric
#     eigvals = np.linalg.eigvalsh(g_matrix)
#     lambda_max = np.max(eigvals)
#     lambda_min = np.min(eigvals) # Just for monitoring singularity
    
#     # D. Compute the Conditional Lipschitz Constant L(theta)
#     # L(theta) = L_euc + (K / epsilon) * ||grad||
#     L_theta = L_EUC + (K_CONST / epsilon) * grad_norm
    
#     # E. Calculate Safe Step Size eta
#     # eta < 2 * eps^2 * (lambda_max + eps) / L(theta)
#     # We use a safety factor of 0.9 to be strictly strictly below the bound
#     eta = 0.9 * (2 * (epsilon**2) * (lambda_max + epsilon)) / L_theta
    
#     # F. QNG Update
#     # theta_new = theta - eta * (g + eps*I)^-1 * grad
#     regularized_metric = g_matrix + epsilon * np.eye(NUM_PARAMS)
#     inv_metric = np.linalg.inv(regularized_metric)
    
#     # Flatten grad for matrix multiplication
#     grad_flat = np.reshape(grad, (NUM_PARAMS,))
#     step = eta * (inv_metric @ grad_flat)
    
#     # Reshape step back to parameter shape
#     step_reshaped = np.reshape(step, SHAPE)
#     theta_new = theta - step_reshaped
    
#     # Logging
#     if current_step_idx % 5 == 0:
#         print(f"Step {current_step_idx}: Cost {cost_fn(theta):.4f}")
#         print(f"  > Grad Norm: {grad_norm:.4f}")
#         print(f"  > Metric Lambda_min: {lambda_min:.6f} (Singular if close to 0)")
#         print(f"  > Computed L(theta): {L_theta:.4f}")
#         print(f"  > Adaptive Step Size eta: {eta:.6f}")
        
#     return theta_new

# # --- 4. EXECUTION ---

# # Initialize parameters
# np.random.seed(42)
# theta = np.random.random(SHAPE, requires_grad=True)

# print(f"Starting Optimization...")
# print(f"Euclidean Constant L_Euc: {L_EUC:.4f}")
# print(f"Riemannian Constant K: {K_CONST:.4f}")
# print("-" * 30)

# # Optimization Loop
# epsilon = 1  # Fixed regularization parameter

# for t in range(30):
#     theta = adaptive_qng_step(theta, t, epsilon=epsilon)

# print("-" * 30)
# print(f"Final Cost: {cost_fn(theta):.4f}")

import numpy as np
import matplotlib.pyplot as plt

def simulate_deviation(K, steps=100, dt=0.01):
    """
    Simulates the scalar Jacobi equation: J'' + K*J = 0
    J represents the deviation/error between two nearby paths.
    """
    t = np.linspace(0, steps*dt, steps)
    J = np.zeros(steps)
    
    # Initial Conditions: Small deviation, no initial divergence speed
    J[0] = 0.1 
    J_dot = 0.0
    
    for i in range(steps - 1):
        # acceleration = -K * position
        J_ddot = -K * J[i]
        
        # Euler integration
        J_dot += J_ddot * dt
        J[i+1] = J[i] + J_dot * dt
        
    return t, J

# --- Simulation Parameters based on Paper's Scaling ---
# Low P (Small circuit): Curvature K is small
# High P (Deep circuit): Curvature K scales as P^4 (massive)

params = {
    'Low Curvature (Small P)': 1.0,      # K ~ 1
    'Med Curvature (Med P)': 10.0,       # K ~ 10
    'High Curvature (Large P)': 100.0    # K ~ 100 (Simulating the P^4 effect)
}

plt.figure(figsize=(10, 6))

for label, K in params.items():
    t, J = simulate_deviation(K, steps=200, dt=0.02)
    plt.plot(t, J, label=f"{label} (K={K})", linewidth=2)

plt.title("The 'Reality' of High Curvature: Geodesic Deviation")
plt.xlabel("Optimization Steps (Time)")
plt.ylabel("Deviation Magnitude ||J|| (Instability)")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Visualization created.")