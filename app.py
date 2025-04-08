import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from light_generator import LightPValueGenerator
from light_statistics import LightPValueStatistics
from cokepepsi_strategy import CokePepsiTradingStrategy

# Set page configuration
st.set_page_config(
    page_title="P-Value Analysis & Trading Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize our classes
generator = LightPValueGenerator()
stats = LightPValueStatistics()

# Initialize trading strategy
@st.cache_resource
def load_strategy():
    # Use the data/dataset path primarily
    data_path = "data/dataset/CokePepsi.csv"
    
    try:
        # Try to load the strategy from the main path
        return CokePepsiTradingStrategy(data_path)
    except (FileNotFoundError, ValueError) as e:
        # Fall back to alternative paths if main path fails
        fallback_paths = [
            "data/datasets/CokePepsi.csv",
            "data/data/datasets/CokePepsi.csv",
            "/mount/src/pvalue_analysis/data/dataset/CokePepsi.csv"
        ]
        
        for path in fallback_paths:
            try:
                return CokePepsiTradingStrategy(path)
            except (FileNotFoundError, ValueError):
                continue
        
        # If all paths fail, show error and return None
        st.error(f"Failed to load CokePepsi data. Error: {e}")
        return None

strategy = load_strategy()

# Add custom CSS for better performance
st.markdown("""
<style>
    .main {
        max-width: 1200px;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4e8df5;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("P-Value Analysis: Key Concepts from Taleb's Paper")
st.markdown("""
This application demonstrates key concepts from Nassim Nicholas Taleb's paper on p-values and applies them to trading strategy development:

1. **Meta-Distribution of P-Values**: How p-values vary across statistically identical phenomena
2. **P-Value Hacking**: How multiple trials can lead to false "significant" results
3. **Trading Strategy Application**: How these concepts apply to trading strategy development and evaluation

These concepts highlight fundamental issues with p-value interpretation in scientific research and financial strategy development.
""")

# Create tabs for different concepts
tabs = st.tabs([
    "Meta-Distribution of P-Values", 
    "P-Value Hacking", 
    "Price Data Analysis", 
    "Strategy Backtest", 
    "P-Value Meta-Distribution in Trading", 
    "P-Hacking in Trading"
])

# Tab 1: Meta-Distribution of P-Values
with tabs[0]:
    st.header("Meta-Distribution of P-Values")
    
    st.markdown("""
    ### The Problem with P-Values
    
    A key insight from Taleb's paper is that p-values themselves have a distribution. Even when the true p-value is 0.05:
    - 75% of realizations will have p < 0.05
    - 25% of realizations will have p > 0.05
    
    This means that p-values are extremely unstable and can vary dramatically across identical experiments.
    """)
    
    # Interactive controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        pM = st.slider("True Median P-Value (pM)", 0.01, 0.5, 0.05, 0.01, 
                      help="The 'true' median p-value in the population")
        
        n_samples = st.slider("Number of Samples", 100, 1000, 500, 100,
                             help="Number of p-value samples to generate")
        
        show_density = st.checkbox("Show Theoretical Density", value=True,
                                  help="Display the theoretical probability density function")
        
        st.markdown(f"""
        ### Key Statistics
        
        - **Probability p < 0.05**: {stats.probability_below_threshold(pM, 0.05):.2f}
        - **Probability p < 0.01**: {stats.probability_below_threshold(pM, 0.01):.2f}
        """)
    
    with col2:
        # Generate data
        p_values = generator.generate_meta_distribution_samples(pM, num_samples=n_samples)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(p_values, bins=30, density=True, alpha=0.7, color='skyblue', 
                label=f'Simulated p-values (n={n_samples})')
        
        # Plot theoretical density if requested
        if show_density:
            p_range = np.linspace(0.001, 0.999, 1000)
            pdf_values = stats.meta_distribution_pdf(p_range, pM)
            ax.plot(p_range, pdf_values, 'r-', linewidth=2, label='Theoretical density')
        
        # Add vertical line at p=0.05
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='p=0.05')
        
        # Add vertical line at p=pM
        ax.axvline(x=pM, color='purple', linestyle='-', alpha=0.7, label=f'True median p={pM}')
        
        # Set labels and title
        ax.set_xlabel('p-value')
        ax.set_ylabel('Density')
        ax.set_title(f'Meta-Distribution of P-Values (True Median p={pM})')
        ax.legend()
        
        # Set x-axis to log scale for better visualization of small p-values
        ax.set_xscale('log')
        
        # Display the plot
        st.pyplot(fig)
        
        st.markdown("""
        ### Interpretation
        
        This visualization shows how p-values are distributed when the true median p-value is fixed. 
        The extreme skewness means that p-values are highly unstable and can vary dramatically 
        across identical experiments.
        
        **Key insight**: Even when the true p-value is 0.05, about 75% of experiments will show p < 0.05,
        leading to potential false positives.
        """)

# Tab 2: P-Value Hacking
with tabs[1]:
    st.header("P-Value Hacking")
    
    st.markdown("""
    ### Multiple Testing and P-Hacking
    
    When researchers perform multiple tests (or "trials") and select the best result,
    they engage in p-hacking. This dramatically increases the chance of finding a 
    "statistically significant" result even when no true effect exists.
    """)
    
    # Interactive controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        pM_hack = st.slider("True Median P-Value", 0.01, 0.5, 0.05, 0.01, key="pM_hack",
                           help="The 'true' median p-value in the population")
        
        max_trials = st.slider("Maximum Number of Trials", 1, 20, 10, 1,
                              help="Maximum number of trials to simulate")
        
        significance = st.slider("Significance Level (α)", 0.001, 0.1, 0.05, 0.001, format="%.3f",
                                help="Threshold for statistical significance")
        
        # Calculate and display statistics
        trials_list = list(range(1, max_trials + 1))
        expected_mins = {m: stats.expected_min_p_value(pM_hack, m) for m in trials_list}
        
        st.markdown("### Expected Minimum P-Values")
        
        # Create a small table of expected minimum p-values
        data = {"Trials": trials_list, "Expected Min P-Value": [expected_mins[m] for m in trials_list]}
        df = pd.DataFrame(data)
        df["Significant at α={}?".format(significance)] = df["Expected Min P-Value"] < significance
        
        # Highlight the first row where the expected min p-value becomes significant
        first_sig = df[df["Significant at α={}?".format(significance)]]["Trials"].min() if any(df["Significant at α={}?".format(significance)]) else None
        
        if first_sig:
            st.markdown(f"**Finding**: With just **{first_sig} trials**, the expected minimum p-value becomes significant at α={significance}.")
        else:
            st.markdown(f"**Finding**: Even with {max_trials} trials, the expected minimum p-value does not become significant at α={significance}.")
        
        st.dataframe(df.style.format({"Expected Min P-Value": "{:.4f}"}))
    
    with col2:
        # Generate data for visualization
        min_p_values = []
        for m in range(1, max_trials + 1):
            min_p_values.append(generator.generate_p_hacking_samples(pM_hack, m_trials=m, num_experiments=1000))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create boxplots
        ax.boxplot(min_p_values, tick_labels=range(1, max_trials + 1))
        
        # Add horizontal line at significance level
        ax.axhline(y=significance, color='red', linestyle='--', alpha=0.7, 
                   label=f'Significance level (α={significance})')
        
        # Add line for expected minimum p-values
        ax.plot(range(1, max_trials + 1), [expected_mins[m] for m in range(1, max_trials + 1)], 
                'go-', linewidth=2, label='Expected minimum p-value')
        
        # Set labels and title
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Minimum P-Value')
        ax.set_title('Distribution of Minimum P-Values Across Multiple Trials')
        ax.legend()
        
        # Set y-axis to log scale for better visualization
        ax.set_yscale('log')
        
        # Display the plot
        st.pyplot(fig)
        
        st.markdown("""
        ### Interpretation
        
        This visualization shows how the minimum p-value decreases as researchers perform more trials.
        The boxplots show the distribution of minimum p-values across 1000 simulated experiments.
        
        **Key insight**: With multiple trials, researchers can easily obtain "statistically significant" 
        results even when no true effect exists. This is why p-hacking is so problematic in research.
        """)

# Tab 3: Price Data Analysis
with tabs[2]:
    st.header("Price Data Analysis")
    
    if strategy is None:
        st.error("⚠️ Unable to load Coke-Pepsi data. Please check the data paths and file format.")
    else:
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
    
    if strategy is not None:
        st.markdown("""
        ### Spread Analysis
        
        The pairs trading strategy is based on the spread between KO and PEP prices, adjusted by a hedge ratio.
        When the spread deviates significantly from its mean, it signals a potential trading opportunity.
        """)
        
        # Calculate and display the spread
        window_size = st.slider("Window Size for Spread Calculation", 30, 120, 60, 5)
        try:
            spread_df = strategy.calculate_spread(window_size=window_size)
        except Exception as e:
            st.error(f"Error calculating spread: {e}")
            spread_df = None
    
    if strategy is not None and spread_df is not None:
        try:
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
        except Exception as e:
            st.error(f"Error plotting spread data: {e}")
    
    if strategy is not None and spread_df is not None:
        try:
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
        except Exception as e:
            st.error(f"Error plotting hedge ratio and p-value data: {e}")

# Tab 4: Strategy Backtest
with tabs[3]:
    st.header("Pairs Trading Strategy Backtest")
    
    if strategy is None:
        st.error("⚠️ Unable to load Coke-Pepsi data. Please check the data paths and file format.")
    else:
        st.markdown("""
        This section demonstrates a simple pairs trading strategy based on the spread between KO and PEP.
        
        **Strategy Rules:**
        1. When the spread's z-score exceeds the entry threshold, enter a position (long the underperformer, short the outperformer)
        2. When the spread's z-score returns within the exit threshold, exit the position
        3. Only enter trades when the p-value of the regression is below the significance threshold
        """)
    
    if strategy is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Strategy Parameters")
            
            backtest_window = st.slider("Window Size", 30, 120, 60, 5, key="backtest_window")
            entry_threshold = st.slider("Entry Threshold (Z-Score)", 1.0, 3.0, 2.0, 0.1)
            exit_threshold = st.slider("Exit Threshold (Z-Score)", 0.1, 1.5, 0.5, 0.1)
            p_value_threshold = st.slider("P-Value Threshold", 0.001, 0.1, 0.05, 0.001, format="%.3f", key="p_value_threshold_backtest")
            
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
    
    if strategy is not None:
        with col2:
            st.subheader("Strategy Results")
            
            if hasattr(strategy, 'strategy_results') and len(strategy.strategy_results) > 0:
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

# Tab 5: P-Value Meta-Distribution in Trading
with tabs[4]:
    st.header("P-Value Meta-Distribution in Trading")
    
    if strategy is None:
        st.error("⚠️ Unable to load Coke-Pepsi data. Please check the data paths and file format.")
    else:    
        st.markdown("""
        This section demonstrates how p-values can vary dramatically across statistically identical trading strategies.
        We generate multiple bootstrap samples from the data and calculate the p-value for each sample.
        
        **Key Insights:**
        - Even when the true relationship between assets exists, p-values can be extremely unstable
        - This instability leads to potential false positives and false negatives in trading strategy development
        - The meta-distribution of p-values is highly skewed, especially for small p-values
        """)
    
    if strategy is not None:
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
    
    if strategy is not None:
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

# Tab 6: P-Hacking in Trading
with tabs[5]:
    st.header("P-Hacking Risk in Trading Strategy Development")
    
    if strategy is None:
        st.error("⚠️ Unable to load Coke-Pepsi data. Please check the data paths and file format.")
    else:
        st.markdown("""
        This section demonstrates how testing multiple parameter combinations can lead to false "significant" trading strategies (p-hacking).
        
        **Key Insights:**
        - Even when no true edge exists, testing multiple parameters will likely yield "successful" strategies by chance
        - The more parameter combinations tested, the higher the probability of finding a seemingly profitable strategy
        - This phenomenon is directly related to the meta-distribution of p-values and multiple testing issues
        """)
    
    if strategy is not None:
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
    
    if strategy is not None:
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

# Footer with references
st.markdown("---")
st.markdown("""
### References

Taleb, N.N. (2019). *The Meta-Distribution of Standard P-Values*. 
[https://arxiv.org/abs/1603.07532](https://arxiv.org/abs/1603.07532)

This application demonstrates key concepts from Taleb's paper and their implications for trading strategy development.
""")

# Add a sidebar about the app
st.sidebar.markdown("### About This App")
st.sidebar.info("""
This application demonstrates key concepts from Nassim Nicholas Taleb's paper on p-values and their application to trading strategies.

The app includes:
1. **General P-Value Concepts**:
   - Meta-distribution of p-values
   - P-value hacking effects
   
2. **Trading Strategy Application**:
   - Pairs trading strategy for Coca-Cola and PepsiCo
   - Demonstration of p-value instability in financial time series
   - Visualization of p-hacking risks in strategy development

Use the tabs to explore different aspects of the analysis.
""")

st.sidebar.markdown("### Key Takeaways")
st.sidebar.markdown("""
1. P-values are extremely unstable and can vary dramatically across identical tests
2. Testing multiple parameter combinations will likely yield "successful" strategies by chance
3. Out-of-sample testing is crucial to validate whether a strategy has a true edge
4. Even strategies with high Sharpe ratios in backtesting may fail in live trading due to p-hacking effects
""")

st.sidebar.markdown("### Loading Time")
st.sidebar.info("Loading time: < 10 seconds for the basic app. Complex analyses like p-hacking evaluation may take longer depending on the number of parameter combinations tested.")