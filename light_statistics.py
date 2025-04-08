import numpy as np
from scipy.stats import t, norm
from scipy.special import erfc, erfcinv, erf, erfinv

class LightPValueStatistics:
    """
    Lightweight class implementing core statistical functions for p-value analysis
    based on Taleb's paper, focusing only on meta-distribution and p-value hacking.
    """
    
    def __init__(self):
        """Initialize the LightPValueStatistics class."""
        pass
    
    def meta_distribution_pdf(self, p, pM, n=20):
        """
        Calculate the probability density function (PDF) of the meta-distribution of p-values.
        Simplified implementation focusing on the limiting distribution for better performance.
        
        Args:
            p (float or array): p-value(s) to calculate the PDF for
            pM (float): The true median p-value
            n (int): Sample size (default=20)
            
        Returns:
            float or array: PDF value(s) at p
        """
        # Handle array input
        if isinstance(p, (list, np.ndarray)):
            return np.array([self.meta_distribution_pdf(p_i, pM, n) for p_i in p])
        
        # For p = 0.5, the distribution is uniform
        if abs(p - 0.5) < 1e-10:
            return 1.0
        
        # For very small p, use the approximation
        if p < 0.01:
            return self.small_p_approximation(p, pM)
        
        # Calculate the PDF using the limiting distribution formula
        try:
            erfc_inv_2p = erfcinv(2*p)
            erfc_inv_2pM = erfcinv(2*pM)
            
            return np.exp(erfc_inv_2pM * (erfc_inv_2pM - 2*erfc_inv_2p))
        except (ValueError, OverflowError):
            # Handle numerical issues
            return 0.0
    
    def small_p_approximation(self, p, pM):
        """
        Calculate the approximation for small p-values.
        
        Args:
            p (float): p-value to calculate the approximation for
            pM (float): The true median p-value
            
        Returns:
            float: Approximated PDF value at p
        """
        # Avoid numerical issues
        if p <= 0 or pM <= 0:
            return 0.0
        
        try:
            term1 = np.sqrt(2*pM)
            term2 = np.sqrt(np.log(1/(2*pM**2)))
            term3 = np.exp(-(np.log(np.log(1/(2*p**2))) - 2*np.log(p))**2 / 
                           (np.log(np.log(1/(2*pM**2))) - 2*np.log(pM)))
            
            return term1 * term2 * term3
        except (ValueError, OverflowError, ZeroDivisionError):
            # Handle numerical issues
            return 0.0
    
    def probability_below_threshold(self, pM, threshold=0.05):
        """
        Calculate the probability that a p-value from the meta-distribution is below a threshold.
        
        Args:
            pM (float): The true median p-value
            threshold (float): p-value threshold (e.g., 0.05)
            
        Returns:
            float: Probability that a p-value is below the threshold
        """
        # Generate samples to estimate the probability
        # This is a simplified approach for better performance
        z_pM = norm.ppf(1 - pM)
        z_threshold = norm.ppf(1 - threshold)
        
        # Calculate probability using normal distribution properties
        prob = norm.cdf(z_threshold - z_pM)
        
        return prob
    
    def expected_min_p_value(self, pM, m_trials):
        """
        Estimate the expected minimum p-value across m trials.
        Simplified implementation for better performance.
        
        Args:
            pM (float): The true median p-value
            m_trials (int): Number of trials
            
        Returns:
            float: Expected minimum p-value
        """
        # For a single trial, the expected p-value is just pM
        if m_trials == 1:
            return pM
        
        # For multiple trials, use a simplified approximation
        # Based on order statistics of uniform distribution
        # This is much faster than Monte Carlo simulation
        return pM / (m_trials + 1)
