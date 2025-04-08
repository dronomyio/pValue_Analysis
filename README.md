# P-Value Analysis Tool

## Overview

This application demonstrates key concepts from Nassim Nicholas Taleb's paper on p-values, particularly focusing on:

1. **Meta-Distribution of P-Values**: How p-values vary across statistically identical phenomena
2. **P-Value Hacking**: How multiple trials can lead to false "significant" results
3. **Application to Trading Strategies**: Practical implications for financial analysis

## Mathematical Foundation

Based on the source "pvalue_taleb.pdf", the probability distribution function (PDF) for a sample-derived one-tailed p-value (P) from a paired T-test with an unknown variance, a median value M(P) = p_M, and a sample size n, across an ensemble of statistically identical copies of the sample, is given by Proposition 1: 

The PDF formula is: 

If \( p \geq \frac{1}{2} \): 

\[ \phi(p; p_M )_H = (1 - \lambda'_p) \cdot \frac{1}{2^{-n-1}} \cdot ((\lambda'_p - 1) \cdot (\lambda_{p_M} - 1) / \lambda'_p \cdot (-\lambda_{p_M}) + 2 \cdot \sqrt{(1- \lambda'_p) \cdot \lambda'_p \cdot \sqrt{1- \lambda_{p_M}} \cdot \lambda_{p_M} + 1})^{(n+1)/2} \] 

Where: 

\[ \lambda_p = I^{-1}_{2p}(n/2, 1/2) \] 

\[ \lambda_{p_M} = I^{-1}(1-2p_M)(1/2, n/2) \] 

\[ \lambda'_p = I^{-1}(2p-1)(1/2, n/2) \] 

With \( I^{-1}(.) \) being the inverse beta regularized function. 

## Generating Synthetic Data

To generate synthetic data from this distribution, typically follow these steps using computational tools: 

1. Choose values for the parameters: Specify the sample size n and the median p-value p_M (where 0 <= p_M <= 1). 
2. Understand the inverse beta regularized function: The core of the PDF involves the inverse beta regularized function, requiring a computational library that can evaluate this function. 
3. Implement a sampling method: Sampling directly from this complex PDF may be challenging. Common methods for generating random numbers from a given distribution include: 
   - Inverse Transform Sampling: If the cumulative distribution function (CDF) can be derived and inverted, generate a uniform random number u and find p such that CDF(p) = u. Deriving the CDF from this PDF is complex and not explicitly provided in the source. 
   - Acceptance-Rejection Method: Requires knowing a simpler distribution that bounds the target distribution. Generate random numbers from the simpler distribution and accept or reject them based on the ratio of the target and simpler distribution's densities. This requires careful selection of the bounding distribution. 
   - Numerical Approximation: Numerically approximate the PDF and potentially use methods tailored for discrete distributions if the range of p-values is discretized. 

Given the complexity of the PDF, the most practical approach for generating synthetic data would likely involve statistical software providing functions for the beta distribution and numerical integration or root-finding methods to handle the inverse beta regularized function indirectly. 

## Visualizing the Distribution

To draw the diagram of this distribution: 

1. Choose the parameters n and p_M: For example, choose n = 10 and p_M = 0.15 as indicated in Figure 1. 
2. Evaluate the PDF over a range of p-values: Calculate phi(p; p_M) for numerous p values between 0 and 1 using the formulas for phi(p; pM )L for p < 0.5 and phi(p; pM )H for p > 0.5, accurately evaluating the inverse beta regularized functions involved. 
3. Plot the results: Create a graph with the p-values on the x-axis and the corresponding PDF values phi(p; p_M) on the y-axis to show the shape of the distribution. 

The source provides examples of such diagrams in Figure 2, showing the distribution's convergence for different values of n, and Figure 3, illustrating the probability distribution of a one-tailed p-value with a specific expected value and its skewness. Figure 4 demonstrates the distribution for different values of p_M, indicating how p_M = 0.5 leads to a uniform distribution, as mentioned in Remark 1. 

## Key Observations

Key observations from the source about the distribution: 

- The distribution of p-values is extremely skewed (right-skewed). This means that for a given "true" p-value, observed p-values across different experiments can vary greatly, with a higher likelihood of observing smaller p-values than the true one. 
- The distribution is volatile and varies significantly across repetitions of identical protocols. 
- The skewness can lead to illusions of "statistical significance" because the average p-value can be considerably higher than most individual observations. 
- For p_M = 1/2, the distribution converges to a uniform distribution. 
- As the sample size n becomes large, the distribution converges to a limiting form given in Proposition 2. 

## Creating Synthetic Datasets

Creating a synthetic dataset based on Proposition 1: 

Proposition 1 in the source "pvalue_taleb.pdf" describes the probability distribution function (PDF) for a sample-derived one-tailed p-value (P) from the paired T-test statistic (unknown variance) with a median value M(P) = p_M derived from a sample of size n. Key characteristics include extreme skewness, dependence on sample size (n), and dependence on median p-value (p_M). 

To create synthetic data for this proposition, follow these conceptual steps using computational tools: 

1. Choose Parameter Values: Decide on specific values for the sample size n and the median p-value p_M (e.g., p_M = 0.15 as considered in Figure 1). 
2. Implement the Inverse Beta Regularized Function: The formulas for phi(p; pM )L and phi(p; pM )H rely on the inverse beta regularized function, requiring computational support. 
3. Implement the PDF: Use the formulas provided to implement the piecewise PDF. 
4. Choose a Sampling Method: Practical approaches for generating synthetic data might involve: 
   - Acceptance-Rejection Method: Finding a simpler "envelope" distribution majorizing the target PDF.
   - Numerical Inversion (Approximate Inverse Transform Sampling): Numerically approximate the CDF by integrating the PDF and then numerically find the inverse of the CDF. 
5. Generate random numbers using your chosen sampling method and the implemented PDF. These numbers represent synthetic p-values drawn from the distribution described by Proposition 1 for your chosen n and p_M. 

## Application to Trading Strategies

Application development using the concepts from "pvalue_taleb.pdf": 

Drawing on the information from "pvalue_taleb.pdf", you could develop an application that evaluates the robustness and potential for spurious findings in trading strategies based on statistical significance observed in historical stock data. The paper emphasizes the skewness and volatility of p-values across ensembles of statistically identical phenomena, highlighting the dangers of "p-hacking". 

To integrate these concepts with stock data: 

1. **Backtesting Trading Strategies Across Multiple Simulated Histories**: Define a trading strategy relying on some testable hypothesis related to stock price movements. Instead of backtesting on a single historical period, simulate multiple statistically identical market histories using techniques like bootstrapping. 
2. **Analyzing the Meta-Distribution of P-values**: Collect the distribution of p-values from simulated backtests. Analyze this meta-distribution by visualizing the histogram of p-values, calculating the median p-value (p_M), and comparing the observed distribution to theoretical forms described in the paper. 
3. **Assessing the Risk of Spurious Significance**: Highlight the probability of observing low p-values across simulations, even if the underlying strategy has no real edge. This concept relates to the issue of "p-hacking". Estimate the expected minimum p-value from multiple variations of a trading strategy. 
4. **Applying Insights on Significance Levels**: Explore implications of stricter significance levels to reduce false positives in trading strategy evaluation. Illustrate how increasing sample size affects the stability and distribution of p-values. 

Such an application helps users understand financial market randomness and the limitations of relying solely on statistical significance from a single backtest. It directly applies concepts from "pvalue_taleb.pdf" to quantitative finance, aiding in avoiding pitfalls of "p-hacked" trading strategies.

## Running the Application

To run the application:

```bash
streamlit run app.py
```

## Reference

Taleb, N.N. (2019). *The Meta-Distribution of Standard P-Values*. 
[https://arxiv.org/abs/1603.07532](https://arxiv.org/abs/1603.07532)