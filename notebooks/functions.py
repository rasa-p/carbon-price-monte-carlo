"""
Shared functions for the carbon price Monte Carlo project.

Contains only function definitions (no code that runs on import), so any
script can safely do:

    from functions import calculate_var_es, bootstrap_var_es, confidence_level_analysis

SIGN CONVENTION: these use the LOSS convention throughout.
VaR and ES are returned as POSITIVE numbers representing a loss.
A P&L of -20 EUR corresponds to a VaR of +20 EUR.
"""

import numpy as np


def calculate_var_es(pnl_samples, confidence_level):
    """
    Calculate VaR and ES for a given set of profit and loss samples.

    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (float): Confidence level for VaR and ES calculation (eg. 0.95 for 95% confidence).

    Returns:
    tuple: VaR and ES values.
    """

    var = -np.quantile(pnl_samples, 1-confidence_level)

    es = -np.mean(pnl_samples[pnl_samples <= -var])

    return var, es


def bootstrap_var_es(pnl_samples, confidence_level, n_bootstrap=1000):
    """
    Bootstrap VaR and ES to estimate error.

    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (float): Confidence level for VaR and ES calculation (eg. 0.95 for 95% confidence).
    n_bootstrap (int): Number of bootstrap samples.

    Returns:
    tuple: Arrays of bootstrapped VaR and ES values.
    """
    var_bootstrap, es_bootstrap = np.zeros(n_bootstrap), np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        resampled_pnls = np.random.choice(pnl_samples, size = len(pnl_samples), replace = True)
        var_bootstrap[i], es_bootstrap[i] = calculate_var_es(resampled_pnls, confidence_level)

    return np.std(var_bootstrap), np.std(es_bootstrap)


def confidence_level_analysis(pnl_samples, confidence_level, n_bootstrap=1000):
    """
    Analyze VaR and ES for different confidence levels and estimate error.

    Parameters:
    pnl_samples (array): Array of profit and loss samples.
    confidence_level (list): List of confidence levels.
    n_bootstrap (int): Number of bootstrap samples.

    Returns:
    dict: Dictionary containing VaR, ES, and their errors for each confidence level.
    """
    results = {}
    for cl in confidence_level:
        var, es = calculate_var_es(pnl_samples, cl)
        var_error, es_error = bootstrap_var_es(pnl_samples, cl, n_bootstrap)
        results[cl] = {
            "VaR": var,
            "ES": es,
            "VaR_Error": var_error,
            "ES_Error": es_error
        }
    return results
