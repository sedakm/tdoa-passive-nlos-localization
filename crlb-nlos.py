import numpy as np

# -----------------------------
# Geometry
# -----------------------------

tx = np.array([0,0])

receivers = np.array([
    [-500,0],
    [500,0],
    [0,500],
    [0,-500]
])

true_position = np.array([200,150])

# -----------------------------
# Noise parameters
# -----------------------------

sigma_los = 2
sigma_b = 5

p_los = 0.7
p_nlos = 0.3

# -----------------------------
# Equivalent variance
# -----------------------------

sigma_eq2 = p_los*sigma_los**2 + p_nlos*(sigma_los**2 + sigma_b**2)

# -----------------------------
# Jacobian computation
# -----------------------------

def compute_jacobian(x):

    H = []

    for rx in receivers:

        d_tx = np.linalg.norm(x - tx)
        d_rx = np.linalg.norm(x - rx)

        dx = (x[0]-tx[0])/d_tx + (x[0]-rx[0])/d_rx
        dy = (x[1]-tx[1])/d_tx + (x[1]-rx[1])/d_rx

        H.append([dx,dy])

    return np.array(H)

H = compute_jacobian(true_position)

# -----------------------------
# Fisher Information Matrix
# -----------------------------

FIM = (1/sigma_eq2) * H.T @ H

# -----------------------------
# CRLB
# -----------------------------

CRLB = np.linalg.inv(FIM)

# -----------------------------
# Position Error Bound
# -----------------------------

PEB = np.sqrt(np.trace(CRLB))

print("Fisher Information Matrix:")
print(FIM)

print("\nCRLB Covariance:")
print(CRLB)

print("\nPosition Error Bound (meters):")
print(PEB)