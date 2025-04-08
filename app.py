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
        try:
            prob_05_val = stats.probability_below_threshold(pM, 0.05)
            prob_01_val = stats.probability_below_threshold(pM, 0.01)
            
            if isinstance(prob_05_val, (int, float)):
                prob_05 = str(round(prob_05_val, 2))
            else:
                prob_05 = str(prob_05_val)
                
            if isinstance(prob_01_val, (int, float)):
                prob_01 = str(round(prob_01_val, 2))
            else:
                prob_01 = str(prob_01_val)
        except Exception as e:
            prob_05 = "Error calculating"
            prob_01 = "Error calculating"
            st.error("Error calculating probabilities: " + str(e))
        
        st.markdown("""
        ### Key Statistics
        
        - **Probability p < 0.05**: """ + prob_05 + """
        - **Probability p < 0.01**: """ + prob_01 + """
        """)
    
    with col2:
        # Generate data
        p_values = generator.generate_meta_distribution_samples(pM, num_samples=n_samples)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(p_values, bins=30, density=True, alpha=0.7, color='skyblue', 
                label='Simulated p-values (n=' + str(n_samples) + ')')
        
        # Plot theoretical density if requested
        if show_density:
            p_range = np.linspace(0.001, 0.999, 1000)
            pdf_values = stats.meta_distribution_pdf(p_range, pM)
            ax.plot(p_range, pdf_values, 'r-', linewidth=2, label='Theoretical density')
        
        # Add vertical line at p=0.05
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='p=0.05')
        
        # Add vertical line at p=pM
        ax.axvline(x=pM, color='purple', linestyle='-', alpha=0.7, label='True median p=' + str(pM))
        
        # Set labels and title
        ax.set_xlabel('p-value')
        ax.set_ylabel('Density')
        ax.set_title('Meta-Distribution of P-Values (True Median p=' + str(pM) + ')')
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
        significant_col = "Significant at α=" + str(significance) + "?"
        df[significant_col] = df["Expected Min P-Value"] < significance
        
        # Highlight the first row where the expected min p-value becomes significant
        first_sig = df[df[significant_col]]["Trials"].min() if any(df[significant_col]) else None
        
        if first_sig:
            st.markdown("**Finding**: With just **" + str(first_sig) + " trials**, the expected minimum p-value becomes significant at α=" + str(significance))
        else:
            st.markdown("**Finding**: Even with " + str(max_trials) + " trials, the expected minimum p-value does not become significant at α=" + str(significance))
        
        # Use a simpler formatting approach without format specifiers
        st.dataframe(df)
    
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
                   label='Significance level (α=' + str(significance) + ')')
        
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
                        st.write("File saved to: " + data_path)
                    except Exception as file_error:
                        st.error("Error writing file: " + str(file_error))
                    
                    data_found = True
                    try:
                        file_size_bytes = len(bytes_data)
                        if isinstance(file_size_bytes, (int, float)):
                            file_size_kb = round(file_size_bytes/1024, 1)
                        else:
                            file_size_kb = file_size_bytes/1024
                        st.success("File uploaded and validated successfully! Size: " + str(file_size_kb) + " KB")
                    except Exception as e:
                        st.success("File uploaded and validated successfully!")
                else:
                    st.error("The CSV file needs at least 2 columns for KO and PEP prices.")
            except Exception as e:
                st.error("Error uploading file: " + str(e))
    
    with data_source_tabs[1]:
        st.write("Use the included sample data for demonstration:")
        if st.button("Load Sample Data"):
            # Use the verified path we found
            sample_path = "data/datasets/CokePepsi.csv"
            
            if os.path.exists(sample_path):
                try:
                    st.write("Debug: Found sample data at: " + sample_path)
                    
                    # Create directory if it doesn't exist
                    target_dir = os.path.dirname(data_path)
                    os.makedirs(target_dir, exist_ok=True)
                    st.write("Debug: Created directory: " + target_dir)
                    
                    # Copy the file
                    import shutil
                    shutil.copy(sample_path, data_path)
                    st.write("Debug: File copied from: " + sample_path + " to: " + data_path)
                    
                    # Verify the file was copied
                    if os.path.exists(data_path):
                        st.write("Debug: Verified file exists at: " + data_path + " with size: " + str(os.path.getsize(data_path)) + " bytes")
                        data_found = True
                        st.success("Sample data loaded successfully!")
                    else:
                        st.error("File copy failed. The destination file does not exist.")
                except Exception as e:
                    st.error("Error loading sample data: " + str(e))
                    import traceback
                    st.code(traceback.format_exc(), language="python")
            else:
                st.error("Sample data file not found at: " + sample_path)
                
                # Try alternate locations as backup
                backup_paths = [
                    "data/data/datasets/CokePepsi.csv",
                    "data/CokePepsi.csv"
                ]
                
                for backup_path in backup_paths:
                    if os.path.exists(backup_path):
                        st.write("Found backup sample data at: " + backup_path)
                        try:
                            import shutil
                            shutil.copy(backup_path, data_path)
                            data_found = True
                            st.success("Sample data loaded from backup location!")
                            break
                        except Exception as e:
                            st.error("Error loading from backup: " + str(e))
                
                if not data_found:
                    st.error("Could not find any sample data files. Please use the upload option instead.")
    
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
                    st.write("File copied from: " + custom_path + " to: " + data_path)
                    data_found = True
                    st.success("Data loaded successfully from " + custom_path + "!")
                except Exception as e:
                    st.error("Error loading data from path: " + str(e))
            else:
                st.error("File not found at the specified path.")
    
    if not data_found:
        st.warning("""
        CokePepsi.csv data file was not found. Please upload the file using the file uploader above.
        
        The file should contain adjusted closing prices for Coca-Cola (KO) and PepsiCo (PEP) stocks.
        """)
    else:
        # Debug section - before anything else
        st.write("Debug: Starting data processing")
        # Load and display data
        try:
            st.write("Debug: Loading data from " + data_path)
            
            # Check if the file exists
            if not os.path.exists(data_path):
                st.error("File does not exist at path: " + data_path)
                st.error("Current directory: " + os.getcwd())
                df = pd.DataFrame()  # Empty dataframe to avoid errors
            else:
                st.write("Debug: File exists with size: " + str(os.path.getsize(data_path)) + " bytes")
                # Try to read the file contents
                try:
                    with open(data_path, 'r') as f:
                        file_preview = f.read(500)  # Read first 500 chars
                        st.write("Debug: File preview:\n" + file_preview)
                except Exception as read_error:
                    st.error("Error reading file contents: " + str(read_error))
                
                # Now try to parse as CSV
                df = pd.read_csv(data_path)
                st.write("Debug: Data loaded successfully with shape " + str(df.shape))
            
            # Convert data to numeric to ensure we have numbers, not strings
            # Only convert data if we have a non-empty dataframe
            if df.empty:
                st.error("DataFrame is empty. Skipping data conversion.")
            else:
                # Convert data to numeric to ensure we have numbers, not strings
                st.write("Debug: Converting data to numeric")
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                st.write("Debug: Data columns converted to numeric types")
                
                # Check if all data became NaN after conversion
                if df.isnull().all().all():
                    st.error("All data converted to NaN. Check that the CSV contains numeric data.")
                    # Show the original data for debugging
                    st.write("Original data before conversion:")
                    original_df = pd.read_csv(data_path)
                    st.write(original_df.head())
            
            # Add row numbers as an index approximating time
            df_with_index = df.copy()
            df_with_index['Day'] = range(len(df_with_index))
            st.write("Debug: Added day index")
            
            # Create tabs for different visualizations
            data_tabs = st.tabs(["Data Preview", "Price Series", "Normalized Prices", "Spread Analysis"])
            
            with data_tabs[0]:
                st.subheader("CokePepsi.csv Data Preview")
                
                # Check if dataframe is empty
                if df.empty:
                    st.error("DataFrame is empty! No data to display.")
                else:
                    st.write("Debug: Displaying dataframe with shape: " + str(df.shape))
                    st.write("Debug: DataFrame columns: " + str(list(df.columns)))
                    st.write("Debug: First 5 rows (raw display):")
                    st.write(df.head(5))  # Use st.write for raw display
                    
                    # Also try standard dataframe display
                    st.write("Standard dataframe display:")
                    st.dataframe(df.head(10))
                
                # Print data statistics
                st.subheader("Data Statistics")
                st.write("**Rows:** " + str(len(df)) + " trading days")
                st.write("**Columns:** " + ', '.join(df.columns))
                try:
                    # Ensure values are numeric before rounding
                    min_ko_val = df.iloc[:, 0].min()
                    max_ko_val = df.iloc[:, 0].max()
                    if isinstance(min_ko_val, (int, float)) and isinstance(max_ko_val, (int, float)):
                        min_ko = round(min_ko_val, 2)
                        max_ko = round(max_ko_val, 2)
                    else:
                        min_ko = min_ko_val
                        max_ko = max_ko_val
                    st.write("**KO Price Range:** $" + str(min_ko) + " to $" + str(max_ko))
                    
                    min_pep_val = df.iloc[:, 1].min()
                    max_pep_val = df.iloc[:, 1].max()
                    if isinstance(min_pep_val, (int, float)) and isinstance(max_pep_val, (int, float)):
                        min_pep = round(min_pep_val, 2)
                        max_pep = round(max_pep_val, 2)
                    else:
                        min_pep = min_pep_val
                        max_pep = max_pep_val
                    st.write("**PEP Price Range:** $" + str(min_pep) + " to $" + str(max_pep))
                except Exception as e:
                    st.write("Error calculating price range: " + str(e))
                
                # Calculate correlation
                try:
                    correlation = df.iloc[:, 0].corr(df.iloc[:, 1])
                    if isinstance(correlation, (int, float)):
                        corr_rounded = round(correlation, 3)
                    else:
                        corr_rounded = correlation
                    st.write("**Correlation:** " + str(corr_rounded))
                except Exception as e:
                    st.write("Error calculating correlation: " + str(e))
            
            with data_tabs[1]:
                try:
                    # Plot the data with better formatting
                    st.subheader("Price Series")
                    st.write("Debug: Entering Price Series tab")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    st.write("Debug: Created figure")
                    
                    # Check data dimensions before plotting
                    st.write("Debug: Checking data dimensions for plotting")
                    # Check dimensions and skip plotting if insufficient
                    data_ok_for_plotting = True
                    if df_with_index.shape[1] < 2:
                        st.error("Data must have at least 2 columns for Price Series visualization")
                        st.write("Current data shape: " + str(df_with_index.shape))
                        st.write("Available columns: " + str(list(df_with_index.columns)))
                        data_ok_for_plotting = False
                        
                    # Ensure column labels exist
                    if len(df.columns) < 2:
                        col1_label = "Column 1"
                        col2_label = "Column 2"
                    else:
                        col1_label = str(df.columns[0])
                        col2_label = str(df.columns[1])
                    
                    # Only plot if data is valid
                    if data_ok_for_plotting:
                        # Plot with better formatting
                        ax.plot(df_with_index['Day'], df_with_index.iloc[:, 0], label=col1_label, color='blue', linewidth=2)
                        ax.set_ylabel("Coca-Cola Price ($)", color='blue')  # Hardcoded label instead of using format
                        
                        # Create a second y-axis for PEP
                        ax2 = ax.twinx()
                        ax2.plot(df_with_index['Day'], df_with_index.iloc[:, 1], label=col2_label, color='red', linewidth=2)
                    else:
                        # Display error message on plot
                        ax.text(0.5, 0.5, "Insufficient data for visualization", 
                               horizontalalignment='center', verticalalignment='center',
                               transform=ax.transAxes, fontsize=14)
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
                    
                except Exception as e:
                    st.error("Error in Price Series tab: " + str(e))
                
                st.markdown("""
                **Observations:**
                - Both KO and PEP prices generally move together over time
                - PEP trades at a higher price than KO 
                - There are periods where the stocks diverge, creating trading opportunities
                """)
            
            with data_tabs[2]:
                try:
                    # Plot normalized prices
                    st.subheader("Normalized Price Series")
                    st.write("Debug: Entering Normalized Price Series tab")
                    
                    # Check data dimensions before normalizing
                    st.write("Debug: About to normalize data")
                    # Check dimensions and skip plotting if insufficient
                    data_ok_for_normalization = True
                    if df.shape[1] < 2 or df.shape[0] == 0:
                        st.error("Data must have at least 2 columns and 1 row for Normalized Price visualization")
                        st.write("Current data shape: " + str(df.shape))
                        st.write("Available columns: " + str(list(df.columns)))
                        data_ok_for_normalization = False
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Only proceed with normalization if data is valid
                    if data_ok_for_normalization:
                        try:
                            # Ensure first row has valid numeric data for division
                            first_row = df.iloc[0]
                            st.write("Debug: First row values: " + str(first_row.values))
                            
                            # Check for zeros or NaNs in first row
                            if first_row.isnull().any() or (first_row == 0).any():
                                st.warning("First row contains zeros or NaN values, using careful normalization")
                                # Use a safer normalization approach
                                normalized = df.copy()
                                for col in df.columns:
                                    base_val = df[col].iloc[0]
                                    if base_val != 0 and not pd.isna(base_val):
                                        normalized[col] = df[col] / base_val * 100
                                    else:
                                        normalized[col] = 100  # Default to flat line
                            else:
                                # Normal normalization
                                normalized = df / df.iloc[0] * 100
                            
                            st.write("Debug: Normalized data shape: " + str(normalized.shape))
                            
                            # Ensure column labels exist
                            if len(df.columns) < 2:
                                col1_label = "Column 1"
                                col2_label = "Column 2"
                            else:
                                col1_label = str(df.columns[0])
                                col2_label = str(df.columns[1])
                            
                            # Plot the normalized data
                            ax.plot(df_with_index['Day'], normalized.iloc[:, 0], label=col1_label, color='blue', linewidth=2)
                            ax.plot(df_with_index['Day'], normalized.iloc[:, 1], label=col2_label, color='red', linewidth=2)
                            
                        except Exception as norm_error:
                            st.error("Error normalizing data: " + str(norm_error))
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            
                            # Display error message on plot
                            ax.text(0.5, 0.5, "Error in data normalization", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=14)
                    else:
                        # Display error message on plot if invalid data
                        ax.text(0.5, 0.5, "Insufficient data for normalization", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=14)
                    
                    # Add title and labels
                    ax.set_title("Normalized Prices (First day = 100)")
                    ax.set_xlabel("Trading Day")
                    ax.set_ylabel("Normalized Price")
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error("Error in Normalized Prices tab: " + str(e))
                
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
                try:
                    # Calculate a simple spread and z-score for demonstration
                    st.subheader("Spread Analysis")
                    st.write("Debug: Entering Spread Analysis tab")
                    
                    # Check data dimensions before calculating spread
                    st.write("Debug: Checking data dimensions for spread analysis")
                    data_ok_for_spread = True
                    if df.shape[1] < 2 or df.shape[0] == 0:
                        st.error("Data must have at least 2 columns and 1 row for Spread Analysis")
                        st.write("Current data shape: " + str(df.shape))
                        st.write("Available columns: " + str(list(df.columns)))
                        data_ok_for_spread = False
                    
                    # Only calculate ratio if data is valid
                    if data_ok_for_spread:
                        # Calculate ratio with error checking
                        df_spread = df.copy()
                        
                        # Check for zeros in the denominator
                        if (df_spread.iloc[:, 1] == 0).any() or df_spread.iloc[:, 1].isnull().any():
                            st.warning("Second column contains zeros or NaN values, replacing with small values for ratio calculation")
                            # Replace zeros and NaNs with a small value to avoid division by zero
                            df_spread.iloc[:, 1] = df_spread.iloc[:, 1].replace(0, 0.0001)
                            df_spread.iloc[:, 1] = df_spread.iloc[:, 1].fillna(0.0001)
                        
                        df_spread['Ratio'] = df_spread.iloc[:, 0] / df_spread.iloc[:, 1]
                        st.write("Debug: Calculated ratio")
                    else:
                        # Create an empty dataframe with Ratio column for consistency
                        df_spread = pd.DataFrame(columns=['Ratio'])
                    
                    # Only proceed with spread analysis if data is valid
                    if data_ok_for_spread:
                        # Calculate z-score with a window that adjusts to data length
                        # Make sure window size is appropriate for data length
                        if len(df_spread) < 60:
                            window = max(5, len(df_spread) // 4)  # Use at least 5 days or 1/4 of data length
                            st.warning(f"Data length ({len(df_spread)}) is less than default window (60). Using window size of {window} instead.")
                        else:
                            window = 60
                            
                        # Check if data is sufficient for window analysis
                        data_sufficient_for_window = len(df_spread) > window
                        if not data_sufficient_for_window:
                            st.error(f"Data length ({len(df_spread)}) is too short for rolling window analysis with window size {window}.")
                    else:
                        # Set window value for when data is not OK
                        window = 60
                        data_sufficient_for_window = False
                        
                    # Create figure first to ensure we always have something to display
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                    
                    # Check if we should skip further analysis
                    if not data_ok_for_spread or not data_sufficient_for_window:
                        # Display error message on plot
                        ax1.text(0.5, 0.5, "Not enough data for spread analysis", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax1.transAxes, fontsize=14)
                        ax2.text(0.5, 0.5, "Need more than " + str(window) + " data points", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax2.transAxes, fontsize=14)
                        plt.tight_layout()
                        # Skip the rest of the analysis
                    
                    # Only proceed with z-score calculation if data is valid and sufficient
                    if data_ok_for_spread and data_sufficient_for_window:
                        try:
                            df_spread['Ratio_MA'] = df_spread['Ratio'].rolling(window=window).mean()
                            df_spread['Ratio_SD'] = df_spread['Ratio'].rolling(window=window).std()
                            
                            # Handle division by zero in z-score calculation
                            df_spread['Z-Score'] = 0  # Initialize with zeros
                            mask = df_spread['Ratio_SD'] > 0  # Only calculate where SD > 0
                            df_spread.loc[mask, 'Z-Score'] = (df_spread.loc[mask, 'Ratio'] - df_spread.loc[mask, 'Ratio_MA']) / df_spread.loc[mask, 'Ratio_SD']
                            
                            st.write("Debug: Calculated z-score")
                            
                            # Drop NaN values from the beginning
                            df_spread = df_spread.dropna()
                            
                            # Add index for plotting
                            df_spread['Day'] = range(len(df_spread))
                            
                            st.write("Debug: Created figure")
                            
                            # Plot the ratio
                            ax1.plot(df_spread['Day'], df_spread['Ratio'], label='KO/PEP Ratio')
                            ax1.plot(df_spread['Day'], df_spread['Ratio_MA'], label=str(window) + '-day Moving Average', color='red')
                            ax1.set_title('Ratio of KO to PEP Prices')
                            ax1.set_ylabel('Price Ratio')
                            ax1.grid(True, alpha=0.3)
                            ax1.legend()
                            st.write("Debug: Plotted ratio")
                            
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
                            st.write("Debug: Plotted z-score")
                        except Exception as z_error:
                            st.error("Error calculating z-score: " + str(z_error))
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            
                            # Display error message on plot
                            ax1.text(0.5, 0.5, "Error in z-score calculation", 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=ax1.transAxes, fontsize=14)
                            ax2.text(0.5, 0.5, str(z_error), 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=ax2.transAxes, fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error("Error in Spread Analysis tab: " + str(e))
                
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
            error_msg = str(e)
            st.error("Error loading or displaying data: " + error_msg)
            
            # Special handling for specific errors
            if "Unknown format code" in error_msg:
                st.error("String formatting error detected. This is often caused by an f-string or format issue.")
                st.info("Please contact the developer with this error message.")
            elif "doesn't define round method" in error_msg:
                st.error("Type error with round() function. This is caused by trying to round a non-numeric value.")
                st.info("Try a different data file or report this issue to the developer.")
                
                # Print stack trace for debugging (remove in production)
                import traceback
                st.code(traceback.format_exc(), language="python")

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