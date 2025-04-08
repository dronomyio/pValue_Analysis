import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cokepepsi_strategy import CokePepsiTradingStrategy

# Set page configuration
st.set_page_config(
    page_title="Coke-Pepsi Trading Strategy with P-Value Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Coke-Pepsi Trading Strategy with P-Value Meta-Distribution Analysis")
st.markdown("""
This application demonstrates a pairs trading strategy for Coca-Cola (KO) and PepsiCo (PEP) stocks, 
with a focus on the meta-distribution of p-values as described in Taleb's paper.

The key insights from this analysis:
1. P-values are extremely unstable and can vary dramatically across identical experiments
2. Multiple parameter testing can lead to false "significant" trading strategies (p-hacking)
3. Trading strategies that seem profitable may be artifacts of randomness, not true alpha
""")

# Initialize strategy
@st.cache_resource
def load_strategy():
    return CokePepsiTradingStrategy("data/datasets/CokePepsi.csv")

strategy = load_strategy()

# Create tabs for different parts of the analysis
tab1, tab2, tab3, tab4 = st.tabs([
    "Price Data Analysis", 
    "Basic Strategy Backtest", 
    "P-Value Meta-Distribution", 
    "P-Hacking Analysis"
])

# Tab 1: Price Data Analysis
with tab1:
    st.header("Price Data Analysis")
    st.markdown("""
    First, let's examine the price data for Coca-Cola (KO) and PepsiCo (PEP) stocks.
    These two companies are ideal for pairs trading as they operate in the same industry and tend to be affected by similar market factors.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Price Series")
        fig = strategy.plot_price_series()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Normalized Price Series")
        fig = strategy.plot_normalized_prices()
        st.pyplot(fig)
    
    st.markdown("""
    ### Spread Analysis
    
    The pairs trading strategy is based on the spread between KO and PEP prices, adjusted by a hedge ratio.
    When the spread deviates significantly from its mean, it signals a potential trading opportunity.
    """)
    
    # Calculate and display the spread
    window_size = st.slider("Window Size for Spread Calculation", 30, 120, 60, 5)
    spread_df = strategy.calculate_spread(window_size=window_size)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot spread
    ax1.plot(spread_df['spread'], label='Spread')
    ax1.set_title('Spread Between KO and PEP (hedge ratio adjusted)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot z-score
    ax2.plot(spread_df['z_score'], label='Z-Score', color='orange')
    ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Entry Threshold (+2)')
    ax2.axhline(y=-2.0, color='r', linestyle='--', alpha=0.3, label='Entry Threshold (-2)')
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Exit Threshold (+0.5)')
    ax2.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3, label='Exit Threshold (-0.5)')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Z-Score of Spread (normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### Hedge Ratio and P-Value Analysis
    
    The hedge ratio determines the relationship between KO and PEP prices in our pairs trading strategy.
    The p-value indicates the statistical significance of this relationship.
    """)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot hedge ratio
    ax1.plot(spread_df['hedge_ratio'], label='Hedge Ratio')
    ax1.set_title('Hedge Ratio (Beta) Between KO and PEP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot p-value
    ax2.plot(spread_df['p_value'], label='P-Value', color='green')
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.3, label='Significance Threshold (p=0.05)')
    ax2.set_title('P-Value of Regression (lower values indicate stronger relationship)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization of small p-values
    
    plt.tight_layout()
    st.pyplot(fig)

# Tab 2: Basic Strategy Backtest
with tab2:
    st.header("Pairs Trading Strategy Backtest")
    st.markdown("""
    This section demonstrates a simple pairs trading strategy based on the spread between KO and PEP.
    
    **Strategy Rules:**
    1. When the spread's z-score exceeds the entry threshold, enter a position (long the underperformer, short the outperformer)
    2. When the spread's z-score returns within the exit threshold, exit the position
    3. Only enter trades when the p-value of the regression is below the significance threshold
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Strategy Parameters")
        
        backtest_window = st.slider("Window Size", 30, 120, 60, 5)
        entry_threshold = st.slider("Entry Threshold (Z-Score)", 1.0, 3.0, 2.0, 0.1)
        exit_threshold = st.slider("Exit Threshold (Z-Score)", 0.1, 1.5, 0.5, 0.1)
        p_value_threshold = st.slider("P-Value Threshold", 0.001, 0.1, 0.05, 0.001, format="%.3f")
        
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                results, sharpe = strategy.backtest_strategy(
                    window_size=backtest_window,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    p_value_threshold=p_value_threshold
                )
                
                st.success(f"Backtest complete! Sharpe Ratio: {sharpe:.2f}")
                
                # Get the current strategy ID
                strategy_id = f"w{backtest_window}_e{entry_threshold}_x{exit_threshold}_p{p_value_threshold}"
                
                # Get additional metrics
                annual_return = strategy.strategy_results[strategy_id]['annual_return']
                annual_vol = strategy.strategy_results[strategy_id]['annual_vol']
                
                st.metric("Annual Return", f"{annual_return:.2%}")
                st.metric("Annual Volatility", f"{annual_vol:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col2:
        st.subheader("Strategy Results")
        
        if len(strategy.strategy_results) > 0:
            # Get the current strategy ID
            current_strategy_id = f"w{backtest_window}_e{entry_threshold}_x{exit_threshold}_p{p_value_threshold}"
            
            # Check if the current strategy has been backtested
            if current_strategy_id in strategy.strategy_results:
                fig = strategy.plot_strategy_results(current_strategy_id)
                st.pyplot(fig)
            else:
                # If not, use the most recent strategy
                latest_strategy_id = list(strategy.strategy_results.keys())[-1]
                fig = strategy.plot_strategy_results(latest_strategy_id)
                st.pyplot(fig)
                st.info(f"Showing results for strategy with parameters: {strategy.strategy_results[latest_strategy_id]['params']}. Run a backtest with current parameters to update.")
        else:
            st.info("No backtest results yet. Run a backtest to see the results.")

# Tab 3: P-Value Meta-Distribution
with tab3:
    st.header("P-Value Meta-Distribution Analysis")
    st.markdown("""
    This section demonstrates how p-values can vary dramatically across statistically identical phenomena.
    We generate multiple bootstrap samples from the data and calculate the p-value for each sample.
    
    **Key Insights:**
    - Even when the true effect exists, p-values can be extremely unstable
    - This instability leads to potential false positives and false negatives
    - The meta-distribution of p-values is highly skewed, especially for small p-values
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        ensemble_samples = st.slider("Number of Bootstrap Samples", 100, 1000, 500, 100)
        ensemble_window = st.slider("Window Size for Regression", 30, 120, 60, 5, key="ensemble_window")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
        
        if st.button("Generate P-Value Ensembles"):
            with st.spinner("Generating p-value ensembles..."):
                ensembles = strategy.generate_p_value_ensembles(
                    num_samples=ensemble_samples,
                    window_size=ensemble_window,
                    test_size=test_size
                )
                
                st.success(f"Generated {ensemble_samples} p-value samples!")
                
                # Calculate statistics
                train_p_values = ensembles['p_values']
                test_p_values = ensembles['test_p_values']
                
                st.markdown("### P-Value Statistics")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Median Train P-Value", f"{np.median(train_p_values):.4f}")
                    st.metric("Proportion of 'Significant' Train Results (p < 0.05)", f"{np.mean(train_p_values < 0.05):.2%}")
                
                with col_b:
                    st.metric("Median Test P-Value", f"{np.median(test_p_values):.4f}")
                    st.metric("Proportion of 'Significant' Test Results (p < 0.05)", f"{np.mean(test_p_values < 0.05):.2%}")
    
    with col2:
        st.subheader("P-Value Distribution Visualization")
        
        if hasattr(strategy, 'p_value_ensembles') and strategy.p_value_ensembles:
            fig = strategy.plot_p_value_distribution()
            st.pyplot(fig)
            
            st.markdown("""
            ### Interpretation
            
            This visualization shows how p-values are distributed across multiple statistically similar samples.
            The extreme variability demonstrates why relying on a single p-value from a single test can be misleading.
            
            Note the difference between training and test sets:
            - Training p-values are generally lower due to in-sample fitting (p-hacking)
            - Test p-values represent out-of-sample performance and are generally higher (reality check)
            
            The proportion of "significant" results should theoretically match the significance level (e.g., 5% for α=0.05) under the null hypothesis,
            but often exceeds this due to factors like autocorrelation and non-stationarity in financial time series.
            """)
        else:
            st.info("No p-value ensembles generated yet. Click 'Generate P-Value Ensembles' to see the results.")

# Tab 4: P-Hacking Analysis
with tab4:
    st.header("P-Hacking Risk Analysis")
    st.markdown("""
    This section demonstrates how testing multiple parameter combinations can lead to false "significant" trading strategies (p-hacking).
    
    **Key Insights:**
    - Even when no true edge exists, testing multiple parameters will likely yield "successful" strategies by chance
    - The more parameter combinations tested, the higher the probability of finding a seemingly profitable strategy
    - This phenomenon is directly related to the meta-distribution of p-values and multiple testing issues
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        st.markdown("Select parameter ranges to test:")
        
        window_min = st.slider("Minimum Window Size", 20, 60, 30, 5)
        window_max = st.slider("Maximum Window Size", 60, 120, 90, 5)
        window_step = st.slider("Window Size Step", 5, 30, 15, 5)
        window_sizes = list(range(window_min, window_max + 1, window_step))
        
        entry_min = st.slider("Minimum Entry Threshold", 1.0, 2.0, 1.5, 0.1)
        entry_max = st.slider("Maximum Entry Threshold", 2.0, 3.5, 3.0, 0.1)
        entry_step = st.slider("Entry Threshold Step", 0.1, 0.5, 0.5, 0.1)
        entry_thresholds = [round(t, 1) for t in np.arange(entry_min, entry_max + 0.1, entry_step)]
        
        exit_min = st.slider("Minimum Exit Threshold", 0.1, 0.5, 0.5, 0.1)
        exit_max = st.slider("Maximum Exit Threshold", 0.5, 2.0, 1.5, 0.1)
        exit_step = st.slider("Exit Threshold Step", 0.1, 0.5, 0.5, 0.1)
        exit_thresholds = [round(t, 1) for t in np.arange(exit_min, exit_max + 0.1, exit_step)]
        
        p_thresholds = [0.01, 0.05, 0.1]
        
        parameter_ranges = {
            'window_size': window_sizes,
            'entry_threshold': entry_thresholds,
            'exit_threshold': exit_thresholds,
            'p_value_threshold': p_thresholds
        }
        
        # Calculate total number of combinations
        total_combinations = (
            len(window_sizes) * 
            len(entry_thresholds) * 
            len(exit_thresholds) * 
            len(p_thresholds)
        )
        
        st.metric("Total Parameter Combinations", total_combinations)
        
        if total_combinations > 200:
            st.warning(f"Testing {total_combinations} parameter combinations may take a while. Consider reducing the ranges.")
        
        if st.button("Run P-Hacking Analysis"):
            if total_combinations > 0:
                with st.spinner(f"Testing {total_combinations} parameter combinations..."):
                    results_df = strategy.evaluate_p_hacking_risk(parameter_ranges)
                    st.success(f"Analysis complete! Tested {len(results_df)} parameter combinations.")
            else:
                st.error("No parameter combinations to test. Please adjust the parameter ranges.")
    
    with col2:
        st.subheader("P-Hacking Analysis Results")
        
        # Check if p-hacking analysis has been run
        if 'results_df' in locals() and isinstance(results_df, pd.DataFrame) and not results_df.empty:
            fig = strategy.plot_p_hacking_results(results_df)
            st.pyplot(fig)
            
            st.markdown("### Top 5 Strategies by Sharpe Ratio")
            top_strategies = results_df.sort_values('sharpe_ratio', ascending=False).head(5)
            st.dataframe(top_strategies)
            
            # Calculate statistics about the analysis
            successful_strategies = results_df[results_df['sharpe_ratio'] > 1]
            success_rate = len(successful_strategies) / len(results_df)
            
            st.markdown(f"""
            ### P-Hacking Statistics
            
            - **Total Parameter Combinations Tested:** {len(results_df)}
            - **Number of "Successful" Strategies (Sharpe > 1):** {len(successful_strategies)}
            - **Success Rate:** {success_rate:.2%}
            - **Maximum Sharpe Ratio Found:** {results_df['sharpe_ratio'].max():.2f}
            - **Expected Maximum Sharpe by Chance:** Due to multiple testing across {len(results_df)} combinations, finding a high Sharpe ratio by chance is likely.
            - **Expected Minimum P-Value:** With {len(results_df)} trials, the expected minimum p-value would be approximately {1/(len(results_df)+1):.4f}
            """)
            
            st.markdown("""
            ### Interpretation
            
            This analysis demonstrates how testing multiple parameter combinations can lead to "p-hacking" in trading strategy development.
            Even if the true edge of a strategy is minimal or non-existent, testing enough parameter combinations will likely yield seemingly successful strategies by chance.
            
            Key insights:
            - The distribution of Sharpe ratios across parameter combinations shows the range of outcomes possible through parameter optimization
            - Many parameter combinations yield "successful" strategies (Sharpe > 1) purely by chance
            - This is directly analogous to the p-value meta-distribution issue in scientific research
            - Out-of-sample testing is crucial to validate whether a strategy has a true edge
            """)
        else:
            st.info("No p-hacking analysis results yet. Click 'Run P-Hacking Analysis' to see the results.")

# Footer
st.markdown("---")
st.markdown("""
### References

Taleb, N.N. (2019). *The Meta-Distribution of Standard P-Values*.
[https://arxiv.org/abs/1603.07532](https://arxiv.org/abs/1603.07532)

This application demonstrates how the concepts from Taleb's paper apply to trading strategy development and backtesting.
""")

# Add sidebar information
st.sidebar.title("About This Application")
st.sidebar.info("""
This application demonstrates a pairs trading strategy for Coca-Cola (KO) and PepsiCo (PEP) stocks,
with a focus on p-value meta-distribution and p-hacking concepts from Nassim Nicholas Taleb's paper.

Use the tabs to explore different aspects of the analysis:
1. **Price Data Analysis**: Examine the price data and spread between KO and PEP
2. **Basic Strategy Backtest**: Backtest a simple pairs trading strategy
3. **P-Value Meta-Distribution**: Explore how p-values vary across statistically identical samples
4. **P-Hacking Analysis**: See how testing multiple parameter combinations can lead to false "significant" results
""")

st.sidebar.markdown("### Key Takeaways")
st.sidebar.markdown("""
1. P-values are extremely unstable and can vary dramatically across identical tests
2. Testing multiple parameter combinations will likely yield "successful" strategies by chance
3. Out-of-sample testing is crucial to validate whether a strategy has a true edge
4. Even strategies with high Sharpe ratios in backtesting may fail in live trading
""")

st.sidebar.markdown("### Data Source")
st.sidebar.markdown("The data used in this application is from the CokePepsi.csv dataset, containing adjusted closing prices for Coca-Cola (KO) and PepsiCo (PEP) stocks.")