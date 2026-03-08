# Install pyswarm for Particle Swarm Optimization
!pip install pyswarm

import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyswarm import pso

# Ensure these are globally available from previous cells
# tx, receivers, true_target, sigma_los, p_los, p_nlos, bistatic_distance, gaussian

# --- CRLB Calculation Function ---
# This function calculates the Position Error Bound for a given noise level
def calculate_crlb_peb(current_sigma_nlos_for_crlb, current_sigma_los, p_los, p_nlos, true_position_crlb, tx_crlb, receivers_crlb):
    # Assume sigma_b in CRLB formula is equivalent to current_sigma_nlos from likelihood model
    sigma_b_for_crlb = current_sigma_nlos_for_crlb
    sigma_eq2 = p_los * current_sigma_los**2 + p_nlos * (current_sigma_los**2 + sigma_b_for_crlb**2)

    def compute_jacobian(x, tx, receivers):
        H = []
        for rx in receivers:
            d_tx = np.linalg.norm(x - tx)
            d_rx = np.linalg.norm(x - rx)
            # Ensure denominators are not zero if target is at tx or rx
            dx = (x[0] - tx[0]) / (d_tx + 1e-9) + (x[0] - rx[0]) / (d_rx + 1e-9)
            dy = (x[1] - tx[1]) / (d_tx + 1e-9) + (x[1] - rx[0]) / (d_rx + 1e-9)
            H.append([dx, dy])
        return np.array(H)

    H = compute_jacobian(true_position_crlb, tx_crlb, receivers_crlb)
    FIM = (1 / sigma_eq2) * H.T @ H
    try:
        CRLB = np.linalg.inv(FIM)
        PEB = np.sqrt(np.trace(CRLB))
    except np.linalg.LinAlgError:
        PEB = np.nan # Return NaN if FIM is singular
    return PEB

# --- Modified Negative Log-Likelihood Function ---
# This function now takes sigma_nlos and mu_b as arguments
def negative_log_likelihood(x, current_r, current_sigma_nlos, current_mu_b):
    ll = 0
    for i, rx in enumerate(receivers):
        d = bistatic_distance(x, rx)
        e = current_r[i] - d

        # sigma_los and p_los are still global/constant for the simulation
        los = gaussian(e, 0, sigma_los)
        nlos = gaussian(e, current_mu_b, current_sigma_nlos)

        mixture = p_los * los + p_nlos * nlos
        ll += np.log(mixture + 1e-12) # Add small epsilon to prevent log(0) issues
    return -ll


num_simulations_per_noise_level = 1000

# Define different noise levels (sigma_nlos, mu_b)
# This range represents varying NLOS conditions in indoor scenarios
noise_level_params = [
    {'name': 'Low Noise', 'sigma_nlos': 8, 'mu_b': 20},
    {'name': 'Medium-Low Noise', 'sigma_nlos': 15, 'mu_b': 30},
    {'name': 'Medium Noise', 'sigma_nlos': 25, 'mu_b': 40},
    {'name': 'Medium-High Noise', 'sigma_nlos': 35, 'mu_b': 50},
    {'name': 'High Noise', 'sigma_nlos': 45, 'mu_b': 60}
]

# Define the optimization methods to compare
optimization_methods = [
    'Nelder-Mead',
    'BFGS',
    'L-BFGS-B',
    'Powell',
    'COBYLA',
    'Differential_Evolution',
    'PSO' # Particle Swarm Optimization
]

# Data structures to store results
all_results = {
    'noise_level': [],
    'method': [],
    'mean_error': [],
    'median_error': [],
    'std_dev_error': [],
    'min_error': [],
    'max_error': [],
    'success_rate': [],
    'mean_time_ms': [],
    'CRLB_PEB': []
}

errors_for_pdfs = {noise['name']: {method: [] for method in optimization_methods} for noise in noise_level_params}

initial_guess = np.array([0.0, 0.0]) # Initial guess for local optimizers
bounds = [(-100, 100), (-100, 100)] # Bounds for global optimizers (x and y limits)

print(f"Starting Monte Carlo simulation with {len(noise_level_params)} noise levels and {num_simulations_per_noise_level} runs per level...")

for noise_idx, noise_info in enumerate(noise_level_params):
    noise_name = noise_info['name']
    current_sigma_nlos = noise_info['sigma_nlos']
    current_mu_b = noise_info['mu_b']

    print(f"\n--- Running for Noise Level: {noise_name} (sigma_nlos={current_sigma_nlos}, mu_b={current_mu_b}) ---")

    # Calculate CRLB for the current noise level
    current_crlb_peb = calculate_crlb_peb(current_sigma_nlos, sigma_los, p_los, p_nlos, true_target, tx, receivers)

    method_errors_current_noise = {method: [] for method in optimization_methods}
    method_times_current_noise = {method: [] for method in optimization_methods}

    for sim_idx in range(num_simulations_per_noise_level):
        # 1. Generate new simulated measurements (r) for this run
        current_r = []
        for rx in receivers:
            d = bistatic_distance(true_target, rx)

            if np.random.rand() < p_los:
                noise = np.random.normal(0, sigma_los)
            else:
                noise = np.random.normal(current_mu_b, current_sigma_nlos) # Use dynamic noise params
            current_r.append(d + noise)
        current_r = np.array(current_r)

        # 2. Apply each optimization method
        for method in optimization_methods:
            start_time = time.time()
            mle_position = None
            success = False

            try:
                if method == 'Differential_Evolution':
                    res = scipy.optimize.differential_evolution(negative_log_likelihood,
                                                                  bounds,
                                                                  args=(current_r, current_sigma_nlos, current_mu_b),
                                                                  disp=False, popsize=15)
                    if res.success: # Differential Evolution uses 'success'
                        mle_position = res.x
                        success = True
                elif method == 'PSO':
                    # PSO requires bounds for x and y
                    lb = [bounds[0][0], bounds[1][0]] # lower bounds
                    ub = [bounds[0][1], bounds[1][1]] # upper bounds
                    # pso returns xopt, fopt
                    xopt, fopt = pso(negative_log_likelihood, lb, ub, args=(current_r, current_sigma_nlos, current_mu_b), swarmsize=100, maxiter=100, disp=False)
                    mle_position = xopt
                    success = True # Assume PSO always finds a solution, though not necessarily global optimum
                else:
                    # Use scipy.optimize.minimize for local optimizers
                    res = scipy.optimize.minimize(negative_log_likelihood,
                                                  initial_guess,
                                                  args=(current_r, current_sigma_nlos, current_mu_b),
                                                  method=method,
                                                  options={'disp': False, 'maxiter': 1000}) # Suppress solver messages
                    if res.success: # minimize uses 'success'
                        mle_position = res.x
                        success = True

                if success and mle_position is not None:
                    error = np.linalg.norm(mle_position - true_target)
                    method_errors_current_noise[method].append(error)
                    errors_for_pdfs[noise_name][method].append(error) # Store for PDFs
                else:
                    method_errors_current_noise[method].append(np.nan)

            except Exception as e:
                # print(f"  Sim {sim_idx + 1}, Method {method} failed: {e}") # Uncomment for detailed debug
                method_errors_current_noise[method].append(np.nan)

            method_times_current_noise[method].append((time.time() - start_time) * 1000) # Store in ms

    # Aggregate results for current noise level
    for method in optimization_methods:
        valid_errors = [e for e in method_errors_current_noise[method] if not np.isnan(e)]
        valid_times = [t for t in method_times_current_noise[method] if not np.isnan(t)]

        all_results['noise_level'].append(noise_name)
        all_results['method'].append(method)
        all_results['CRLB_PEB'].append(current_crlb_peb)

        if valid_errors:
            all_results['mean_error'].append(np.mean(valid_errors))
            all_results['median_error'].append(np.median(valid_errors))
            all_results['std_dev_error'].append(np.std(valid_errors))
            all_results['min_error'].append(np.min(valid_errors))
            all_results['max_error'].append(np.max(valid_errors))
            all_results['success_rate'].append(len(valid_errors) / num_simulations_per_noise_level * 100)
        else:
            all_results['mean_error'].append(np.nan)
            all_results['median_error'].append(np.nan)
            all_results['std_dev_error'].append(np.nan)
            all_results['min_error'].append(np.nan)
            all_results['max_error'].append(np.nan)
            all_results['success_rate'].append(0.0)

        if valid_times:
            all_results['mean_time_ms'].append(np.mean(valid_times))
        else:
            all_results['mean_time_ms'].append(np.nan)

# --- Display Results Table ---
results_df = pd.DataFrame(all_results)
print("\n--- Monte Carlo Simulation Results ---")
display(results_df.round(2))

# --- Plot RMSE vs. Measurement Noise ---
plt.figure(figsize=(12, 7))

for method in optimization_methods:
    method_df = results_df[results_df['method'] == method]
    plt.plot(method_df['noise_level'], method_df['mean_error'], marker='o', label=f'{method} RMSE')

# Plot CRLB
crlb_df = results_df[['noise_level', 'CRLB_PEB']].drop_duplicates().sort_values(by='noise_level', key=lambda x: [noise_level_params.index(next(item for item in noise_level_params if item['name'] == level)) for level in x])
plt.plot(crlb_df['noise_level'], crlb_df['CRLB_PEB'], marker='X', linestyle='--', color='red', label='CRLB (PEB)')

plt.xlabel('Measurement Noise Level (NLOS sigma/mu_b)')
plt.ylabel('Mean Position Error (RMSE)')
plt.title('RMSE of Optimization Algorithms vs. Measurement Noise Level')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Plot PDFs of Errors for Each Algorithm at Each Noise Level ---
for noise_name, methods_errors in errors_for_pdfs.items():
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Error Distribution (PDF) for {noise_name} Noise Level', fontsize=16)

    num_methods = len(optimization_methods)
    # Determine optimal grid size for subplots
    rows = int(np.ceil(np.sqrt(num_methods)))
    cols = int(np.ceil(num_methods / rows))

    for i, method in enumerate(optimization_methods):
        plt.subplot(rows, cols, i + 1)
        errors = methods_errors[method]
        if errors and not all(np.isnan(e) for e in errors):
            plt.hist(errors, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'{method}')
            plt.xlabel('Position Error (meters)')
            plt.ylabel('Density')
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            plt.text(0.5, 0.5, 'No valid errors', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='red')
            plt.title(f'{method}')
            plt.xlabel('Position Error (meters)')
            plt.ylabel('Density')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()