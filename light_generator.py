import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc, erfcinv, erf, erfinv
from scipy.stats import t, norm

class LightPValueGenerator:
    """
    Lightweight class to generate data for p-value demonstrations based on Taleb's paper.
    Focuses only on the most important concepts: meta-distribution and p-value hacking.
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(seed)
    
    def generate_meta_distribution_samples(self, pM, n=20, num_samples=1000):
        """
        Generate samples from the meta-distribution of p-values.
        
        Args:
            pM (float): The true median p-value
            n (int): Sample size
            num_samples (int): Number of samples to generate
            
        Returns:
            numpy.ndarray: Array of p-value samples
        """
        # Calculate tau (the T-statistic) corresponding to the median p-value
        tau = t.ppf(1 - pM, df=n-1)
        
        # Generate random T-statistics around tau
        t_stats = np.random.standard_t(df=n-1, size=num_samples) + tau
        
        # Convert T-statistics to p-values
        p_values = 1 - t.cdf(t_stats, df=n-1)
        
        return p_values
    
    def generate_p_hacking_samples(self, pM, m_trials=5, num_experiments=1000):
        """
        Generate minimum p-values across multiple trials to demonstrate p-hacking.
        
        Args:
            pM (float): The true median p-value
            m_trials (int): Number of trials per experiment
            num_experiments (int): Number of experiments to run
            
        Returns:
            numpy.ndarray: Array of minimum p-values
        """
        min_p_values = []
        
        for _ in range(num_experiments):
            # Generate m_trials p-values from the meta-distribution
            z_pM = norm.ppf(1 - pM)
            z_samples = np.random.normal(loc=z_pM, scale=1, size=m_trials)
            trial_p_values = 1 - norm.cdf(z_samples)
            
            # Take the minimum p-value (p-hacking)
            min_p_values.append(np.min(trial_p_values))
        
        return np.array(min_p_values)
    
    def calculate_expected_min_p_value(self, pM, m_trials_list=[1, 2, 3, 5, 10, 20]):
        """
        Calculate the expected minimum p-value across different numbers of trials.
        
        Args:
            pM (float): The true median p-value
            m_trials_list (list): List of trial counts to calculate for
            
        Returns:
            dict: Dictionary mapping trial counts to expected minimum p-values
        """
        result = {}
        
        for m_trials in m_trials_list:
            min_p_values = self.generate_p_hacking_samples(pM, m_trials, 1000)
            result[m_trials] = np.mean(min_p_values)
        
        return result
