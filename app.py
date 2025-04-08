import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from light_generator import LightPValueGenerator
from light_statistics import LightPValueStatistics
import os
import io
from pathlib import Path

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
This application demonstrates key concepts from Nassim Nicholas Taleb's paper on p-values:

1. **Meta-Distribution of P-Values**: How p-values vary across statistically identical phenomena
2. **P-Value Hacking**: How multiple trials can lead to false "significant" results

These concepts highlight fundamental issues with p-value interpretation in scientific research.
""")

# Create tabs for different concepts
tabs = st.tabs(["Meta-Distribution of P-Values", "P-Value Hacking", "Trading Strategy"])

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
        
        # Format statistics individually for safer string formatting
        prob_05 = "{:.2f}".format(stats.probability_below_threshold(pM, 0.05))
        prob_01 = "{:.2f}".format(stats.probability_below_threshold(pM, 0.01))
        
        st.markdown("""
        ### Key Statistics
        
        - **Probability p < 0.05**: {}
        - **Probability p < 0.01**: {}
        """.format(prob_05, prob_01))
    
    with col2:
        # Generate data
        p_values = generator.generate_meta_distribution_samples(pM, num_samples=n_samples)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(p_values, bins=30, density=True, alpha=0.7, color='skyblue', 
                label='Simulated p-values (n={})'.format(n_samples))
        
        # Plot theoretical density if requested
        if show_density:
            p_range = np.linspace(0.001, 0.999, 1000)
            pdf_values = stats.meta_distribution_pdf(p_range, pM)
            ax.plot(p_range, pdf_values, 'r-', linewidth=2, label='Theoretical density')
        
        # Add vertical line at p=0.05
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='p=0.05')
        
        # Add vertical line at p=pM
        ax.axvline(x=pM, color='purple', linestyle='-', alpha=0.7, label='True median p={}'.format(pM))
        
        # Set labels and title
        ax.set_xlabel('p-value')
        ax.set_ylabel('Density')
        ax.set_title('Meta-Distribution of P-Values (True Median p={})'.format(pM))
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
            st.markdown("**Finding**: With just **{} trials**, the expected minimum p-value becomes significant at α={}".format(first_sig, significance))
        else:
            st.markdown("**Finding**: Even with {} trials, the expected minimum p-value does not become significant at α={}".format(max_trials, significance))
        
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
                   label='Significance level (α={})'.format(significance))
        
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

# Tab 3: Trading Strategy
with tabs[2]:
    st.header("Trading Strategy Application")
    
    st.markdown("""
    ### Pairs Trading Strategy: Coca-Cola vs PepsiCo
    
    This section demonstrates how the concepts from Taleb's paper apply to trading strategy development and evaluation.
    We implement a pairs trading strategy for Coca-Cola (KO) and PepsiCo (PEP) stocks and analyze the role of p-values in trading.
    
    **What is Pairs Trading?**
    
    Pairs trading is a market-neutral strategy that matches a long position in one stock with a short position in another related stock.
    The strategy is based on the assumption that two historically correlated stocks will revert to their statistical relationship after diverging.
    
    **Our Approach:**
    1. Model the relationship between KO and PEP using linear regression
    2. Calculate the spread between the actual prices and the predicted relationship
    3. When the spread deviates significantly (measured by z-score), we take a position expecting reversion
    4. We use p-values to assess the statistical significance of the relationship
    """)
    
    # File handling section
    st.subheader("Load CokePepsi.csv Data")
    
    # Create multiple tabs for different data loading methods
    data_source_tabs = st.tabs(["Upload File", "Use Sample Data", "Enter File Path"])
    
    data_path = "data/datasets/CokePepsi.csv"
    data_found = os.path.exists(data_path)
    
    with data_source_tabs[0]:
        st.write("Upload a CSV file with Coca-Cola and PepsiCo adjusted prices:")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="The file should have two columns with Coca-Cola and PepsiCo adjusted prices")
        
        if uploaded_file is not None:
            try:
                # Save the uploaded file
                bytes_data = uploaded_file.getvalue()
                
                # First try to read it to validate it's a proper CSV
                df_check = pd.read_csv(io.BytesIO(bytes_data))
                if len(df_check.columns) >= 2:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    st.write("Creating directory: {}".format(os.path.dirname(data_path)))
                    
                    # Write the file
                    try:
                        with open(data_path, "wb") as f:
                            f.write(bytes_data)
                        st.write("File saved to: {}".format(data_path))
                    except Exception as file_error:
                        st.error("Error writing file: {}".format(str(file_error)))
                    
                    data_found = True
                    st.success("File uploaded and validated successfully! Size: {:.1f} KB".format(len(bytes_data)/1024))
                else:
                    st.error("The CSV file needs at least 2 columns for KO and PEP prices.")
            except Exception as e:
                st.error("Error uploading file: {}".format(str(e)))
    
    with data_source_tabs[1]:
        st.write("Use the included sample data for demonstration:")
        if st.button("Load Sample Data"):
            # Try to copy from a known location if it exists
            sample_paths = [
                "data/datasets/CokePepsi.csv",
                "data/data/datasets/CokePepsi.csv",
                "data/CokePepsi.csv"
            ]
            
            for sample_path in sample_paths:
                if os.path.exists(sample_path):
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(data_path), exist_ok=True)
                        st.write("Creating directory: {}".format(os.path.dirname(data_path)))
                        
                        # Copy the file
                        import shutil
                        shutil.copy(sample_path, data_path)
                        st.write("File copied from: {} to: {}".format(sample_path, data_path))
                        data_found = True
                        st.success("Sample data loaded successfully!")
                        break
                    except Exception as e:
                        st.error("Error loading sample data: {}".format(str(e)))
            
            if not data_found:
                st.error("Could not find sample data files. Please use the upload option instead.")
    
    with data_source_tabs[2]:
        st.write("Specify the path to an existing CokePepsi.csv file:")
        custom_path = st.text_input("Enter path to CSV file:", placeholder="e.g., /path/to/CokePepsi.csv")
        
        if custom_path and st.button("Load from Path"):
            if os.path.exists(custom_path):
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    st.write("Creating directory: {}".format(os.path.dirname(data_path)))
                    
                    # Copy the file
                    import shutil
                    shutil.copy(custom_path, data_path)
                    st.write("File copied from: {} to: {}".format(custom_path, data_path))
                    data_found = True
                    st.success("Data loaded successfully from {}!".format(custom_path))
                except Exception as e:
                    st.error("Error loading data from path: {}".format(str(e)))
            else:
                st.error("File not found at the specified path.")
    
    if not data_found:
        st.warning("""
        CokePepsi.csv data file was not found. Please upload the file using the file uploader above.
        
        The file should contain adjusted closing prices for Coca-Cola (KO) and PepsiCo (PEP) stocks.
        """)
    else:
        # Load and display data
        try:
            st.write("Debug: Loading data from {}".format(data_path))
            df = pd.read_csv(data_path)
            st.write("Debug: Data loaded successfully with shape {}".format(df.shape))
            
            # Add row numbers as an index approximating time
            df_with_index = df.copy()
            df_with_index['Day'] = range(len(df_with_index))
            st.write("Debug: Added day index")
            
            # Create tabs for different visualizations
            data_tabs = st.tabs(["Data Preview", "Price Series", "Normalized Prices", "Spread Analysis"])
            
            with data_tabs[0]:
                st.subheader("CokePepsi.csv Data Preview")
                st.dataframe(df.head(10))
                
                # Print data statistics
                st.subheader("Data Statistics")
                st.write("**Rows:** {} trading days".format(len(df)))
                st.write("**Columns:** {}".format(', '.join(df.columns)))
                st.write("**KO Price Range:** $" + "{:.2f}".format(df.iloc[:, 0].min()) + " to $" + "{:.2f}".format(df.iloc[:, 0].max()))
                st.write("**PEP Price Range:** $" + "{:.2f}".format(df.iloc[:, 1].min()) + " to $" + "{:.2f}".format(df.iloc[:, 1].max()))
                
                # Calculate correlation
                correlation = df.iloc[:, 0].corr(df.iloc[:, 1])
                st.write("**Correlation:** " + "{:.3f}".format(correlation))
            
            with data_tabs[1]:
                # Plot the data with better formatting
                st.subheader("Price Series")
                st.write("Debug: Entering Price Series tab")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                st.write("Debug: Created figure")
                
                # Plot with better formatting
                ax.plot(df_with_index['Day'], df_with_index.iloc[:, 0], label=str(df.columns[0]), color='blue', linewidth=2)
                ax.set_ylabel("Coca-Cola Price ($)", color='blue')  # Hardcoded label instead of using format
                
                # Create a second y-axis for PEP
                ax2 = ax.twinx()
                ax2.plot(df_with_index['Day'], df_with_index.iloc[:, 1], label=str(df.columns[1]), color='red', linewidth=2)
                ax2.set_ylabel("PepsiCo Price ($)", color='red')  # Hardcoded label instead of using format
                
                # Add title and grid
                ax.set_title("Coca-Cola vs PepsiCo Price Series")
                ax.grid(True, alpha=0.3)
                
                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Add x-axis label
                ax.set_xlabel("Trading Day")
                
                st.pyplot(fig)
                
                st.markdown("""
                **Observations:**
                - Both KO and PEP prices generally move together over time
                - PEP trades at a higher price than KO 
                - There are periods where the stocks diverge, creating trading opportunities
                """)
            
            with data_tabs[2]:
                # Plot normalized prices
                st.subheader("Normalized Price Series")
                st.write("Debug: Entering Normalized Price Series tab")
                
                # Normalize to the first day
                st.write("Debug: About to normalize data")
                normalized = df / df.iloc[0] * 100
                st.write("Debug: Normalized data shape: {}".format(normalized.shape))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_with_index['Day'], normalized.iloc[:, 0], label=str(df.columns[0]), color='blue', linewidth=2)
                ax.plot(df_with_index['Day'], normalized.iloc[:, 1], label=str(df.columns[1]), color='red', linewidth=2)
                
                # Add title and labels
                ax.set_title("Normalized Prices (First day = 100)")
                ax.set_xlabel("Trading Day")
                ax.set_ylabel("Normalized Price")
                
                # Add grid and legend
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                st.markdown("""
                **Normalized Price Analysis:**
                
                This chart shows the percentage change from the first day. It helps visualize:
                - Relative performance between the two stocks
                - Periods of divergence and convergence
                - Long-term growth comparison
                
                The pairs trading strategy exploits periods when one stock outperforms or underperforms the other
                temporarily, expecting the relationship to revert to the mean.
                """)
            
            with data_tabs[3]:
                # Calculate a simple spread and z-score for demonstration
                st.subheader("Spread Analysis")
                
                # Calculate ratio
                df_spread = df.copy()
                df_spread['Ratio'] = df_spread.iloc[:, 0] / df_spread.iloc[:, 1]
                
                # Calculate z-score with a 60-day window
                window = 60
                df_spread['Ratio_MA'] = df_spread['Ratio'].rolling(window=window).mean()
                df_spread['Ratio_SD'] = df_spread['Ratio'].rolling(window=window).std()
                df_spread['Z-Score'] = (df_spread['Ratio'] - df_spread['Ratio_MA']) / df_spread['Ratio_SD']
                
                # Drop NaN values from the beginning
                df_spread = df_spread.dropna()
                
                # Add index for plotting
                df_spread['Day'] = range(len(df_spread))
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                
                # Plot the ratio
                ax1.plot(df_spread['Day'], df_spread['Ratio'], label='KO/PEP Ratio')
                ax1.plot(df_spread['Day'], df_spread['Ratio_MA'], label='{}-day Moving Average'.format(window), color='red')
                ax1.set_title('Ratio of KO to PEP Prices')
                ax1.set_ylabel('Price Ratio')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot the z-score
                ax2.plot(df_spread['Day'], df_spread['Z-Score'], label='Z-Score', color='green')
                ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (+2)')
                ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (-2)')
                ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit Threshold (+0.5)')
                ax2.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5, label='Exit Threshold (-0.5)')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title('Z-Score of KO/PEP Ratio')
                ax2.set_ylabel('Z-Score')
                ax2.set_xlabel('Trading Day')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Trading Strategy Rules:**
                
                1. **Enter** a position when the z-score exceeds +2 or -2 standard deviations:
                   - If z-score > +2: Short KO and Long PEP (ratio is too high)
                   - If z-score < -2: Long KO and Short PEP (ratio is too low)
                
                2. **Exit** the position when the z-score returns to normal levels:
                   - If position is open and z-score returns to between -0.5 and +0.5
                
                3. **P-Value Analysis**:
                   - We use p-values to determine if the relationship between KO and PEP is statistically significant
                   - Only take trades when the p-value of the regression is below a threshold (typically 0.05)
                   - This helps avoid trading on random noise
                """)
            
            st.markdown("""
            ### P-Value Considerations in Trading Strategies
            
            When developing this pairs trading strategy, the role of p-values is critical:
            
            1. **False Positives**: Even with a threshold of p < 0.05, about 5% of trading signals will be false positives
            
            2. **P-Value Instability**: The meta-distribution of p-values shows that they vary dramatically across statistically identical samples
            
            3. **P-Hacking Risk**: Testing multiple parameter combinations (window sizes, thresholds) leads to finding "significant" strategies by chance
            
            4. **Solution**: Robust out-of-sample testing and awareness of the multiple testing problem are essential
            """)
            
            # Information about the full app
            st.info("""
            ### Extended Trading Strategy Analysis
            
            The comprehensive trading strategy functionality is available in the standalone app:
            ```
            streamlit run run_cokepepsi_analysis.py
            ```
            
            This includes:
            - Complete pairs trading strategy implementation with performance metrics
            - P-value meta-distribution analysis through bootstrap sampling
            - P-hacking risk evaluation by testing multiple parameter combinations
            - Visualization of strategy performance across different parameters
            """)
        except Exception as e:
            st.error("Error loading or displaying data: {}".format(str(e)))

# Footer with references
st.markdown("---")
st.markdown("""
### References

Taleb, N.N. (2019). *The Meta-Distribution of Standard P-Values*. 
[https://arxiv.org/abs/1603.07532](https://arxiv.org/abs/1603.07532)

This application demonstrates key concepts from Taleb's paper.
""")

# Add a sidebar about the app
st.sidebar.markdown("### About This App")
st.sidebar.info("""
This application demonstrates key concepts from Nassim Nicholas Taleb's paper on p-values.

The app includes:
1. **Meta-Distribution of P-Values**: How p-values vary across statistically identical phenomena
2. **P-Value Hacking**: How multiple trials can lead to false "significant" results
3. **Trading Strategy Application**: A simplified interface for pairs trading strategies

For a more comprehensive trading strategy analysis, run the standalone app:
```
streamlit run run_cokepepsi_analysis.py
```
""")

st.sidebar.markdown("### Key Takeaways")
st.sidebar.markdown("""
1. P-values are extremely unstable and can vary dramatically across identical tests
2. Testing multiple parameter combinations will likely yield "successful" strategies by chance
3. Out-of-sample testing is crucial to validate whether a strategy has a true edge
""")