import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go # Import plotly

# ----------------------------
# Geometry
# ----------------------------

tx = np.array([0,0])

receivers = np.array([
    [-50,0],
    [50,0],
    [0,50],
    [0,-50]
])

true_target = np.array([20,15])

# ----------------------------
# Noise parameters
# ----------------------------

sigma_los = 2
sigma_nlos = 8
mu_b = 20

p_los = 0.7
p_nlos = 0.3

# ----------------------------
# Bistatic distance
# ----------------------------

def bistatic_distance(x,rx):

    d_tx = np.linalg.norm(x-tx)
    d_rx = np.linalg.norm(x-rx)

    return d_tx + d_rx


# ----------------------------
# Generate measurements
# ----------------------------

r = []

for rx in receivers:

    d = bistatic_distance(true_target,rx)

    if np.random.rand() < p_los:

        noise = np.random.normal(0,sigma_los)

    else:

        noise = np.random.normal(mu_b,sigma_nlos)

    r.append(d+noise)

r = np.array(r)

# ----------------------------
# Gaussian pdf
# ----------------------------

def gaussian(x,mu,sigma):

    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))


# ----------------------------
# Log likelihood
# ----------------------------

def log_likelihood(x):

    ll = 0

    for i,rx in enumerate(receivers):

        d = bistatic_distance(x,rx)
        e = r[i] - d

        los = gaussian(e,0,sigma_los)
        nlos = gaussian(e,mu_b,sigma_nlos)

        mixture = p_los*los + p_nlos*nlos

        ll += np.log(mixture + 1e-12)

    return ll


# ----------------------------
# Evaluate ML on grid
# ----------------------------

x_vals = np.linspace(-100,100,120)
y_vals = np.linspace(-100,100,120)

X,Y = np.meshgrid(x_vals,y_vals)
Z = np.zeros_like(X)

for i in range(len(x_vals)):
    for j in range(len(y_vals)):

        Z[j,i] = -log_likelihood(np.array([x_vals[i],y_vals[j]]))


# ----------------------------
# 3D Plot (Interactive with Plotly)
# ----------------------------

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

fig.update_layout(title='Interactive 3D Plot of Negative Log Likelihood Function',
                  scene = dict(
                    xaxis_title='x position',
                    yaxis_title='y position',
                    zaxis_title='Negative Log Likelihood'))

fig.show()


# ----------------------------
# 2D Contour Plot (Shaded)
# ----------------------------

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis_r') # Use _r for reverse colormap so lower values are brighter
plt.colorbar(label='Negative Log Likelihood')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Contour Plot of Negative Log Likelihood Function (Shaded)')

# Plot true target position
plt.plot(true_target[0], true_target[1], 'rx', markersize=10, label='True Target')

# Plot MLE position from grid search (if available)
# Re-calculate MLE from grid search to ensure consistency if previous cells were not run together
min_z_index_grid = np.unravel_index(np.argmin(Z), Z.shape)
mle_x_grid = x_vals[min_z_index_grid[1]]
mle_y_grid = y_vals[min_z_index_grid[0]]
plt.plot(mle_x_grid, mle_y_grid, 'wo', markersize=8, label='MLE (Grid Search)', markerfacecolor='none')

# Plot MLE position from optimization (if available, using the last calculated mle_position from Monte Carlo)
if 'mle_position' in locals():
    plt.plot(mle_position[0], mle_position[1], 'g*', markersize=10, label='MLE (Optimization)')

plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 2D Contour Plot (Lines)
# ----------------------------

plt.figure(figsize=(10, 8))
contour_lines = plt.contour(X, Y, Z, levels=20, colors='black', linewidths=0.8) # Using 20 levels for lines
plt.clabel(contour_lines, inline=True, fontsize=8) # Add labels to contour lines
plt.colorbar(label='Negative Log Likelihood')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Contour Plot of Negative Log Likelihood Function (Lines)')

# Plot true target position
plt.plot(true_target[0], true_target[1], 'rx', markersize=10, label='True Target')

# Plot MLE position from grid search
plt.plot(mle_x_grid, mle_y_grid, 'wo', markersize=8, label='MLE (Grid Search)', markerfacecolor='none')

# Plot MLE position from optimization (if available)
if 'mle_position' in locals():
    plt.plot(mle_position[0], mle_position[1], 'g*', markersize=10, label='MLE (Optimization)')

plt.legend()
plt.grid(True)
plt.show()