# Deposit-Size-Analytics
Here’s a **comprehensive README** for your **Deposit Size Analytics** project. It includes instructions, dependencies, and descriptions for all the analyses your script performs. You can save it as `README.md`.

---

# Deposit Size Analytics

## Overview

The **Deposit Size Analytics** project analyzes deposits distributed by account sizes over multiple time periods. It performs a comprehensive analysis including descriptive statistics, trends, share & growth, volatility, survival analysis, network analysis, Monte Carlo simulation, Bayesian analysis, scatterplots, and machine learning clustering.

The project is implemented in **Python**, using popular data science and visualization libraries.

---

## Features

1. **Data Cleaning & Preprocessing**

   * Handles missing values and converts numeric columns.
   * Strips extra spaces from column names.

2. **Descriptive Statistics**

   * Computes mean, median, standard deviation, min/max, and quartiles.
   * Summarizes total deposits by account size.

3. **Trend Analysis**

   * Visualizes deposit trends over time using line charts.
   * Handles multiple years/quarters.

4. **Share & Growth Analysis**

   * Calculates percentage share of each account size.
   * Computes growth rates over time.
   * Computes cumulative deposits per account size.

5. **Correlation Heatmap & Boxplot**

   * Visualizes correlations between time periods.
   * Highlights distributions and outliers in deposits.

6. **Volatility & Outliers**

   * Calculates standard deviation for each account size.
   * Detects outliers using IQR method.

7. **Pie Chart Visualization**

   * Shows the proportional share of deposits per account size.

8. **Survival Analysis**

   * Kaplan-Meier curve for deposit “survival” probabilities.

9. **Network Analysis**

   * Creates a weighted network graph of account sizes based on deposit differences.
   * Compatible with NetworkX 2.5.
   * Shows edge weights between account categories.

10. **Monte Carlo Simulation**

    * Simulates future total deposits using growth factor distributions.
    * Produces histogram of simulation outcomes.

11. **Bayesian Analysis**

    * Computes posterior distribution of deposit proportions using Beta distribution.

12. **Scatterplots & Trend Scatter**

    * Visualizes deposit amounts over time by account size.

13. **KMeans Clustering (Machine Learning)**

    * Clusters account sizes based on deposits over time.
    * Visualizes clusters in 2D scatterplots.

---

## Requirements

* Python 3.8+
* Libraries:

  ```bash
  pip install pandas matplotlib seaborn numpy networkx==2.5 lifelines scikit-learn scipy
  ```

> **Note:** NetworkX 2.5 is required if you want to avoid `random_state` errors in `spring_layout`.

---

## Usage

1. Place your dataset CSV file (e.g., `Table-18  Deposits  distributed by Size of Account.csv`) in a directory.
2. Update the `file_path` variable in the script:

```python
file_path = r"C:\path\to\your\dataset.csv"
```

3. Run the script:

```bash
python main.py
```

4. The script will generate:

   * Bar charts, line charts, pie charts, scatterplots, boxplots
   * Heatmaps and survival curves
   * Monte Carlo and Bayesian analysis charts
   * Network analysis graph
   * K-Means clustering plot
   * Saved as PNG files in the same directory

---

## Output Examples

* `analysis_total_by_size.png` – Bar chart of total deposits by account size
* `analysis_trend_over_time.png` – Line chart of deposit trends
* `analysis_heatmap.png` – Correlation heatmap between periods
* `analysis_boxplot.png` – Boxplot of deposits by time
* `analysis_network.png` – Network graph of account sizes
* `analysis_monte_carlo.png` – Histogram from Monte Carlo simulation
* `analysis_bayesian.png` – Posterior distribution plot
* `analysis_ml_clustering.png` – KMeans clustering visualization

---

## Notes

* Ensure all numeric columns in the dataset are properly formatted.
* Network layout in NetworkX 2.5 is **not reproducible** because `seed` is unsupported.
* For reproducible network layouts, upgrade to **NetworkX 2.6+** and use the `seed` argument.

---

## References

* Dataset Source: [https://data.gov.bd/](https://data.gov.bd/)
* NetworkX: [https://networkx.org/](https://networkx.org/)
* Lifelines (Kaplan-Meier): [https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/)
* Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

---
