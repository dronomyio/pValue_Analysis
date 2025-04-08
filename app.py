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

# Tab 3: Trading Strategy
with tabs[2]:
    st.header("Trading Strategy Application")
    
    st.markdown("""
    ### Coke-Pepsi Trading Strategy
    
    This section demonstrates how the concepts from Taleb's paper apply to trading strategy development and evaluation.
    We implement a pairs trading strategy for Coca-Cola (KO) and PepsiCo (PEP) stocks and analyze:
    
    1. How p-values vary across statistical tests in trading
    2. How multiple parameter testing can lead to false "significant" strategies (p-hacking)
    """)
    
    # Allow users to upload CokePepsi.csv if needed
    uploaded_file = st.file_uploader("Upload CokePepsi.csv file if not already available")
    
    data_path = "data/dataset/CokePepsi.csv"
    data_found = os.path.exists(data_path)
    
    if uploaded_file is not None:
        # Save the uploaded file
        bytes_data = uploaded_file.getvalue()
        st.write(f"File uploaded! Size: {len(bytes_data)} bytes")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Write the file
        with open(data_path, "wb") as f:
            f.write(bytes_data)
        
        data_found = True
    
    if not data_found:
        st.warning("""
        CokePepsi.csv data file was not found. Please upload the file using the file uploader above.
        
        The file should contain adjusted closing prices for Coca-Cola (KO) and PepsiCo (PEP) stocks.
        """)
    else:
        # Load and display data
        try:
            df = pd.read_csv(data_path)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("CokePepsi.csv Data Preview")
                st.dataframe(df.head(10))
                
            with col2:
                # Plot the data
                st.subheader("Price Series")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for col in df.columns:
                    ax.plot(df[col], label=col)
                
                ax.set_title("Coca-Cola vs PepsiCo Prices")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.info("""
            The extended trading strategy functionality is available in the standalone app:
            ```
            streamlit run run_cokepepsi_analysis.py
            ```
            
            This includes:
            - Full pairs trading strategy implementation
            - P-value meta-distribution analysis through bootstrap sampling
            - P-hacking risk evaluation through parameter testing
            """)
        except Exception as e:
            st.error(f"Error loading or displaying data: {e}")

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