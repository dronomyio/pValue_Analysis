import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold

class CokePepsiTradingStrategy:
    """
    A trading strategy class for Coke/Pepsi based on p-value metadistribution concepts.
    Implements a pairs trading approach with robustness testing using p-value ensembles.
    """
    
    def __init__(self, data_path):
        """
        Initialize the trading strategy with data.
        
        Args:
            data_path (str): Path to the CSV file containing Coke/Pepsi data
        """
        self.data = pd.read_csv(data_path)
        self.data.columns = ['KO', 'PEP']  # Rename columns for clarity
        
        # Calculate log returns
        self.returns = self.data.pct_change().dropna()
        self.log_returns = np.log(self.data).diff().dropna()
        
        # Store strategy results
        self.strategy_results = {}
        self.p_value_ensembles = {}
        
    def plot_price_series(self):
        """Plot the original price series for KO and PEP"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['KO'], label='Coca-Cola (KO)')
        plt.plot(self.data['PEP'], label='PepsiCo (PEP)')
        plt.title('Coca-Cola vs PepsiCo Stock Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt
    
    def plot_normalized_prices(self):
        """Plot normalized price series for KO and PEP"""
        normalized = self.data / self.data.iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(normalized['KO'], label='Coca-Cola (KO)')
        plt.plot(normalized['PEP'], label='PepsiCo (PEP)')
        plt.title('Normalized Coca-Cola vs PepsiCo Stock Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt
    
    def calculate_spread(self, window_size=60):
        """
        Calculate the spread between KO and PEP using regression.
        
        Args:
            window_size (int): Size of the rolling window for regression
            
        Returns:
            pd.DataFrame: DataFrame with spread and z-score
        """
        spread_df = pd.DataFrame(index=self.data.index)
        
        # Use rolling window to calculate time-varying hedge ratio
        spread_df['hedge_ratio'] = np.nan
        spread_df['spread'] = np.nan
        spread_df['z_score'] = np.nan
        spread_df['p_value'] = np.nan
        
        for i in range(window_size, len(self.data)):
            # Subset data for the current window
            y = self.data['KO'].iloc[i-window_size:i]
            X = self.data['PEP'].iloc[i-window_size:i]
            X = sm.add_constant(X)
            
            # Fit regression model
            model = OLS(y, X).fit()
            beta = model.params[1]
            alpha = model.params[0]
            p_val = model.pvalues[1]
            
            # Store the hedge ratio and p-value
            spread_df['hedge_ratio'].iloc[i] = beta
            spread_df['p_value'].iloc[i] = p_val
            
            # Calculate the spread
            spread = self.data['KO'].iloc[i] - (alpha + beta * self.data['PEP'].iloc[i])
            spread_df['spread'].iloc[i] = spread
            
            # Calculate z-score for the current window
            historical_spreads = spread_df['spread'].iloc[i-window_size:i].dropna()
            if len(historical_spreads) > 0:
                mean_spread = historical_spreads.mean()
                std_spread = historical_spreads.std()
                if std_spread > 0:
                    spread_df['z_score'].iloc[i] = (spread - mean_spread) / std_spread
        
        return spread_df.dropna()
    
    def generate_signals(self, spread_df, entry_threshold=2.0, exit_threshold=0.5, p_value_threshold=0.05):
        """
        Generate trading signals based on spread z-score and p-value.
        
        Args:
            spread_df (pd.DataFrame): Spread DataFrame from calculate_spread
            entry_threshold (float): Z-score threshold for trade entry
            exit_threshold (float): Z-score threshold for trade exit
            p_value_threshold (float): P-value threshold for valid trades
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        signals = pd.DataFrame(index=spread_df.index)
        signals['z_score'] = spread_df['z_score']
        signals['p_value'] = spread_df['p_value']
        signals['signal'] = 0
        
        # Generate signals based on z-score and p-value
        # 1 = Long KO, Short PEP
        # -1 = Short KO, Long PEP
        # 0 = No position
        
        position = 0
        for i in range(1, len(signals)):
            # Only trade if p-value is significant
            valid_signal = signals['p_value'].iloc[i] < p_value_threshold
            
            if position == 0:  # No position
                if valid_signal and signals['z_score'].iloc[i] < -entry_threshold:
                    # Z-score is significantly negative, spread will likely revert upward
                    # Long KO, Short PEP
                    signals['signal'].iloc[i] = 1
                    position = 1
                elif valid_signal and signals['z_score'].iloc[i] > entry_threshold:
                    # Z-score is significantly positive, spread will likely revert downward
                    # Short KO, Long PEP
                    signals['signal'].iloc[i] = -1
                    position = -1
            
            elif position == 1:  # Long KO, Short PEP
                if signals['z_score'].iloc[i] > -exit_threshold:
                    # Spread has reverted, exit position
                    signals['signal'].iloc[i] = 0
                    position = 0
            
            elif position == -1:  # Short KO, Long PEP
                if signals['z_score'].iloc[i] < exit_threshold:
                    # Spread has reverted, exit position
                    signals['signal'].iloc[i] = 0
                    position = 0
            
            # Carry forward position
            if signals['signal'].iloc[i] == 0:
                signals['signal'].iloc[i] = position
        
        return signals
    
    def backtest_strategy(self, window_size=60, entry_threshold=2.0, exit_threshold=0.5, p_value_threshold=0.05):
        """
        Backtest the pairs trading strategy.
        
        Args:
            window_size (int): Size of the rolling window for regression
            entry_threshold (float): Z-score threshold for trade entry
            exit_threshold (float): Z-score threshold for trade exit
            p_value_threshold (float): P-value threshold for valid trades
            
        Returns:
            tuple: (pd.DataFrame, float) - Strategy results and Sharpe ratio
        """
        # Calculate spread and generate signals
        spread_df = self.calculate_spread(window_size)
        signals = self.generate_signals(spread_df, entry_threshold, exit_threshold, p_value_threshold)
        
        # Create a DataFrame for strategy results
        results = pd.DataFrame(index=signals.index)
        results['KO_return'] = self.returns['KO']
        results['PEP_return'] = self.returns['PEP']
        results['signal'] = signals['signal'].shift(1)  # Use previous day's signal
        results['z_score'] = signals['z_score']
        results['p_value'] = signals['p_value']
        
        # Calculate strategy returns
        results['KO_strategy'] = results['signal'] * results['KO_return']
        results['PEP_strategy'] = -results['signal'] * results['PEP_return']  # Negative sign for short position
        results['strategy_return'] = results['KO_strategy'] + results['PEP_strategy']
        
        # Calculate cumulative returns
        results['cum_strategy'] = (1 + results['strategy_return']).cumprod()
        results['cum_KO'] = (1 + results['KO_return']).cumprod()
        results['cum_PEP'] = (1 + results['PEP_return']).cumprod()
        
        # Calculate performance metrics
        annual_return = results['strategy_return'].mean() * 252
        annual_vol = results['strategy_return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        strategy_id = f"w{window_size}_e{entry_threshold}_x{exit_threshold}_p{p_value_threshold}"
        self.strategy_results[strategy_id] = {
            'results': results,
            'sharpe_ratio': sharpe_ratio,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'params': {
                'window_size': window_size,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'p_value_threshold': p_value_threshold
            }
        }
        
        return results, sharpe_ratio
    
    def plot_strategy_results(self, strategy_id=None):
        """
        Plot strategy results.
        
        Args:
            strategy_id (str): Strategy ID to plot. If None, plots the most recent.
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if strategy_id is None and len(self.strategy_results) > 0:
            strategy_id = list(self.strategy_results.keys())[-1]
        
        if strategy_id not in self.strategy_results:
            raise ValueError(f"Strategy ID {strategy_id} not found in results")
        
        results = self.strategy_results[strategy_id]['results']
        params = self.strategy_results[strategy_id]['params']
        sharpe = self.strategy_results[strategy_id]['sharpe_ratio']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot cumulative returns
        ax1.plot(results['cum_strategy'], label='Strategy', color='green')
        ax1.plot(results['cum_KO'], label='KO', color='blue', alpha=0.5)
        ax1.plot(results['cum_PEP'], label='PEP', color='red', alpha=0.5)
        ax1.set_title(f'Cumulative Returns - Sharpe: {sharpe:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot z-score and signals
        ax2.plot(results['z_score'], label='Z-Score', color='purple')
        ax2.axhline(y=params['entry_threshold'], color='r', linestyle='--', alpha=0.3, label='Entry Threshold')
        ax2.axhline(y=-params['entry_threshold'], color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=params['exit_threshold'], color='g', linestyle='--', alpha=0.3, label='Exit Threshold')
        ax2.axhline(y=-params['exit_threshold'], color='g', linestyle='--', alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Z-Score and Thresholds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot signals and p-values
        ax3.plot(results['p_value'], label='P-Value', color='orange')
        ax3.axhline(y=params['p_value_threshold'], color='r', linestyle='--', alpha=0.8, label='P-Value Threshold')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(results['signal'], label='Signal', color='blue', alpha=0.5, drawstyle='steps-post')
        ax3.set_title('P-Values and Signals')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_p_value_ensembles(self, num_samples=500, window_size=60, test_size=0.3):
        """
        Generate ensembles of p-values to demonstrate the meta-distribution.
        Uses bootstrapping to create statistically identical versions of the data.
        
        Args:
            num_samples (int): Number of bootstrap samples
            window_size (int): Window size for each regression
            test_size (float): Size of the test set as a proportion
            
        Returns:
            dict: Dictionary with p-value ensembles
        """
        # Original data
        ko = self.data['KO'].values
        pep = self.data['PEP'].values
        n = len(ko)
        
        # Results storage
        p_values = []
        test_p_values = []
        r_squared_values = []
        
        # Calculate indices for train/test split
        train_size = int(n * (1 - test_size))
        
        for _ in range(num_samples):
            # Create a bootstrap sample by sampling with replacement
            indices = np.random.choice(train_size - window_size, size=train_size - window_size, replace=True)
            indices = np.sort(indices)  # Preserve time ordering
            
            # Get the bootstrap sample
            ko_sample = ko[indices]
            pep_sample = pep[indices]
            
            # Run regression on the bootstrap sample
            X = sm.add_constant(pep_sample)
            model = OLS(ko_sample, X).fit()
            p_values.append(model.pvalues[1])
            r_squared_values.append(model.rsquared)
            
            # Also calculate test p-value
            ko_test = ko[train_size:train_size+window_size]
            pep_test = pep[train_size:train_size+window_size]
            X_test = sm.add_constant(pep_test)
            test_model = OLS(ko_test, X_test).fit()
            test_p_values.append(test_model.pvalues[1])
            
        # Store results
        self.p_value_ensembles = {
            'p_values': np.array(p_values),
            'test_p_values': np.array(test_p_values),
            'r_squared_values': np.array(r_squared_values),
            'parameters': {
                'num_samples': num_samples,
                'window_size': window_size,
                'test_size': test_size
            }
        }
        
        return self.p_value_ensembles
    
    def plot_p_value_distribution(self):
        """
        Plot the distribution of p-values from the ensemble.
        
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if not self.p_value_ensembles:
            raise ValueError("P-value ensembles not generated. Call generate_p_value_ensembles first.")
        
        p_values = self.p_value_ensembles['p_values']
        test_p_values = self.p_value_ensembles['test_p_values']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram of p-values from bootstrap samples
        ax1.hist(p_values, bins=30, alpha=0.7, color='blue', density=True, label='Training')
        ax1.hist(test_p_values, bins=30, alpha=0.5, color='red', density=True, label='Test')
        ax1.axvline(x=0.05, color='black', linestyle='--', label='p=0.05')
        ax1.set_title('Distribution of P-Values')
        ax1.set_xlabel('P-Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate proportion of significant results
        sig_prop_train = np.mean(p_values < 0.05)
        sig_prop_test = np.mean(test_p_values < 0.05)
        
        # Additional information
        ax2.bar(['Train', 'Test'], [sig_prop_train, sig_prop_test], color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Proportion of "Significant" Results (p < 0.05)')
        ax2.set_ylabel('Proportion')
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.05, color='black', linestyle='--', label='Expected Type I Error Rate')
        
        # Add text annotations
        ax2.text(0, sig_prop_train + 0.05, f"{sig_prop_train:.2%}", ha='center')
        ax2.text(1, sig_prop_test + 0.05, f"{sig_prop_test:.2%}", ha='center')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def evaluate_p_hacking_risk(self, parameter_ranges):
        """
        Evaluate the risk of p-hacking by testing multiple parameters.
        
        Args:
            parameter_ranges (dict): Dictionary with parameter ranges to test
                e.g., {'window_size': [30, 60, 90], 'entry_threshold': [1.5, 2.0, 2.5]}
                
        Returns:
            pd.DataFrame: DataFrame with results for all parameter combinations
        """
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_combinations = list(itertools.product(*[parameter_ranges[param] for param in param_names]))
        
        results = []
        
        # Test each parameter combination
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            
            # Run backtest with the current parameters
            try:
                _, sharpe_ratio = self.backtest_strategy(**params)
                
                params_copy = params.copy()
                params_copy['sharpe_ratio'] = sharpe_ratio
                
                # Get the strategy ID
                strategy_id = f"w{params.get('window_size', 60)}_e{params.get('entry_threshold', 2.0)}_x{params.get('exit_threshold', 0.5)}_p{params.get('p_value_threshold', 0.05)}"
                annual_return = self.strategy_results[strategy_id]['annual_return']
                annual_vol = self.strategy_results[strategy_id]['annual_vol']
                
                params_copy['annual_return'] = annual_return
                params_copy['annual_vol'] = annual_vol
                
                results.append(params_copy)
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Calculate proportion of "successful" strategies (e.g., Sharpe > 1)
        success_rate = np.mean(results_df['sharpe_ratio'] > 1)
        
        # Calculate expected maximum Sharpe ratio by random chance
        max_sharpe = results_df['sharpe_ratio'].max()
        
        print(f"Tested {len(results_df)} parameter combinations")
        print(f"Proportion with Sharpe > 1: {success_rate:.2%}")
        print(f"Maximum Sharpe ratio: {max_sharpe:.2f}")
        print(f"Expected maximum Sharpe by p-hacking: Using {len(results_df)} trials, the expected minimum p-value would be approximately {1/(len(results_df)+1):.4f}")
        
        return results_df
    
    def plot_p_hacking_results(self, results_df):
        """
        Plot the results of the p-hacking evaluation.
        
        Args:
            results_df (pd.DataFrame): Results DataFrame from evaluate_p_hacking_risk
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram of Sharpe ratios
        ax1.hist(results_df['sharpe_ratio'], bins=20, alpha=0.7, color='green')
        ax1.axvline(x=1, color='red', linestyle='--', label='Sharpe = 1')
        ax1.axvline(x=results_df['sharpe_ratio'].max(), color='blue', linestyle='--', label=f'Max Sharpe = {results_df["sharpe_ratio"].max():.2f}')
        ax1.set_title('Distribution of Sharpe Ratios Across Parameter Combinations')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create a scatterplot of returns vs volatility (risk-return tradeoff)
        ax2.scatter(results_df['annual_vol'], results_df['annual_return'], 
                    c=results_df['sharpe_ratio'], cmap='viridis', alpha=0.7)
        
        # Add a colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=results_df['sharpe_ratio'].min(), vmax=results_df['sharpe_ratio'].max())
        sm = ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Sharpe Ratio')
        
        # Add labels and title
        ax2.set_xlabel('Annual Volatility')
        ax2.set_ylabel('Annual Return')
        ax2.set_title('Risk-Return Tradeoff Across Parameter Combinations')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Create strategy instance
    strategy = CokePepsiTradingStrategy("data/datasets/CokePepsi.csv")
    
    # Visualize the data
    strategy.plot_price_series()
    plt.show()
    
    strategy.plot_normalized_prices()
    plt.show()
    
    # Run a backtest with default parameters
    results, sharpe = strategy.backtest_strategy()
    print(f"Default strategy Sharpe ratio: {sharpe:.2f}")
    
    # Plot strategy results
    strategy.plot_strategy_results()
    plt.show()
    
    # Generate and visualize p-value ensembles
    strategy.generate_p_value_ensembles(num_samples=500)
    strategy.plot_p_value_distribution()
    plt.show()
    
    # Evaluate p-hacking risk with multiple parameter combinations
    parameter_ranges = {
        'window_size': [30, 45, 60, 75, 90],
        'entry_threshold': [1.5, 2.0, 2.5, 3.0],
        'exit_threshold': [0.5, 1.0, 1.5],
        'p_value_threshold': [0.01, 0.05, 0.1]
    }
    
    results_df = strategy.evaluate_p_hacking_risk(parameter_ranges)
    
    # Plot p-hacking results
    strategy.plot_p_hacking_results(results_df)
    plt.show()