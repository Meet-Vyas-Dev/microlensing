# --- KMTNet Algorithm ---

# Authors: Atousa Kalantari, Somayeh Khakpash


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import weightedstats as ws

def run_kmtnet_fit(times, fluxes, flux_errors):
    """
    Takes light curve data arrays and returns the final Delta_Chi2 metric.
    """
    # Ensure inputs are numpy arrays
    times = np.asarray(times)
    fluxes = np.asarray(fluxes)
    flux_errors = np.asarray(flux_errors)

    # Filter out NaNs from input data before creating DataFrame
    valid_data_indices = ~np.isnan(times) & ~np.isnan(fluxes) & ~np.isnan(flux_errors) & (flux_errors > 0)
    times, fluxes, flux_errors = times[valid_data_indices], fluxes[valid_data_indices], flux_errors[valid_data_indices]

    if len(times) < 5: # KMTNet fit needs sufficient data points
        return None, None, None # Not enough data for KMTNet fit

    data_df = pd.DataFrame({
        'time': times,
        'flux': fluxes,
        'flux_err': flux_errors
    })

    def Ft_high(t, f_1, f_0, t0, t_eff): # Adjusted order for consistency with minimize
        Q = 1 + ((t - t0) / t_eff)**2
        return np.abs(f_1) * (Q**(-1.0 / 2)) + np.abs(f_0)

    def Ft_low(t, f_1, f_0, t0, t_eff): # Adjusted order for consistency with minimize
        Q = 1 + ((t - t0) / t_eff)**2
        return np.abs(f_1) * (1 - (1 + Q / 2)**-2)**(-1.0 / 2) + np.abs(f_0)

    def chi2_high(f_params, t, flux, flux_err, t0, teff):
        f_1, f_0 = f_params # f_params contains [f_1, f_0]
        model = Ft_high(t, f_1, f_0, t0, teff)
        inv_sigma2 = 1.0 / (flux_err**2)
        return np.sum((flux - model)**2 * inv_sigma2)

    def chi2_low(f_params, t, flux, flux_err, t0, teff):
        f_1, f_0 = f_params
        model = Ft_low(t, f_1, f_0, t0, teff)
        inv_sigma2 = 1.0 / (flux_err**2)
        return np.sum((flux - model)**2 * inv_sigma2)

    # --- Grid Search Setup ---
    teff_min, teff_max = 1, 100
    teff_list, t0_tE_list = [], []
    current_teff = teff_min
    while current_teff <= teff_max:
        teff_list.append(current_teff)
        delta = 1/5 if current_teff < 1 else 1/3
        current_teff *= (1 + delta)

    t0_min, t0_max = np.min(times), np.max(times)
    for teff in teff_list:
        t0_current = t0_min
        while t0_current <= t0_max:
            t0_tE_list.append([t0_current, teff])
            delta = 1/5 if teff < 1 else 1/3
            t0_current += delta * teff

    if not t0_tE_list: return None, None, None

    param1, param2 = [], []
    f_initial = [0.01, 0.99] # Initial guess for f_1, f_0

    for i, (t0_val, teff_val) in enumerate(t0_tE_list):
        # Filter data for the current time interval for efficiency
        df_i = data_df[(data_df['time'] > (t0_val - 7 * teff_val)) & (data_df['time'] < (t0_val + 7 * teff_val))]

        if len(df_i) < 10:
            continue # Skip if not enough data in interval

        # Prepare arguments for minimize (t, flux, flux_err, t0_val, teff_val)
        args = (df_i['time'].values, df_i['flux'].values, df_i['flux_err'].values, t0_val, teff_val)

        try:
            result1 = minimize(chi2_high, f_initial, args=args, method='BFGS')
            # Calculate chi2 over the ENTIRE light curve for Delta_Chi2
            model_diff1 = data_df['flux'].values - Ft_high(data_df['time'].values, result1.x[0], result1.x[1], t0_val, teff_val)
            chi2_all1 = np.sum((model_diff1)**2 * (1.0 / (data_df['flux_err'].values**2)))

            result2 = minimize(chi2_low, f_initial, args=args, method='BFGS')
            model_diff2 = data_df['flux'].values - Ft_low(data_df['time'].values, result2.x[0], result2.x[1], t0_val, teff_val)
            chi2_all2 = np.sum((model_diff2)**2 * (1.0 / (data_df['flux_err'].values**2)))

            param1.append([i, t0_val, teff_val, result1.x[0], result1.x[1], result1.fun,
                            len(df_i), chi2_all1, len(data_df)])
            param2.append([i, t0_val, teff_val, result2.x[0], result2.x[1], result2.fun,
                            len(df_i), chi2_all2, len(data_df)])
        except Exception as e:
            print(f"Warning: KMTNet minimize failed for iteration {i}: {e}")
            continue # Continue to next iteration if fit fails

    if not param1 and not param2: return None, None, None # No successful fits

    # choose_best_params
    min_value1 = min(param1, key=lambda x: x[7])
    min_value2 = min(param2, key=lambda x: x[7])


    if min_value1 < min_value2:
              min_value = min_value1
              param = param1
              F_t = Ft_high
              which_regim = 'high'

    else:
              min_value = min_value2
              param = param2
              F_t = Ft_low
              which_regim = 'low'

    for sublist in param:
        if sublist[7] == min_value[7]:
            parameter = sublist







    chi_mlens = parameter[7]
    t0 = parameter[1]
    t_eff = parameter[2]
    f1 = parameter[3]
    f0 = parameter[4]

    # Linear fit for chi2_linearfit
    data_df_interval = data_df[(data_df['time'] > (t0 - 7 * t_eff)) & (data_df['time'] < (t0 + 7 * t_eff))]

    if len(data_df_interval) == 0:
        # If no data in interval for weighted mean, fallback to overall mean
        mean_flux_interval = np.mean(data_df['flux'].values)
    else:
        # Ensure weights are valid (not zero or inf)
        weights = 1.0 / (data_df_interval['flux_err'].values**2)
        valid_weights_indices = ~np.isinf(weights) & ~np.isnan(weights) & (weights > 0)
        if np.sum(valid_weights_indices) > 0:
            mean_flux_interval = ws.weighted_mean(data_df_interval['flux'].values[valid_weights_indices],
                                                  weights[valid_weights_indices])
        else:
            mean_flux_interval = np.mean(data_df['flux'].values)

    chi2_linearfit = np.sum((data_df['flux'] - mean_flux_interval)**2 / (data_df['flux_err']) ** 2)

    # Handle division by zero for delta_chi_squared
    if chi2_linearfit == 0:
        delta_chi_squared_kmt = 0
    else:
        delta_chi_squared_kmt = (abs(chi_mlens - chi2_linearfit) / chi2_linearfit)

    return delta_chi_squared_kmt, (t0, t_eff, f1, f0), F_t, chi_mlens, chi2_linearfit

