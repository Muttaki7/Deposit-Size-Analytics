import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta

# -------------------- Load Dataset --------------------
file_path = r"C:\Users\Muttaki\Desktop\analysis gov\data-resource_2024_06_24_Table-18  Deposits  distributed by Size of Account.csv"
df = pd.read_csv(file_path)

print("------ Dataset Info ------")
print(f"Dataset Counter: 1")
print(f"Total Records (rows): {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print("Columns:", df.columns.tolist())

# -------------------- Data Cleaning --------------------
df = df.dropna(axis=1, how="all")
df.columns = df.columns.str.strip()
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nCleaned Data Sample:\n", df.head())

# -------------------- Descriptive Statistics --------------------
print("\n------ Descriptive Statistics ------")
print(df.describe())

# Total deposits by account size
total_by_size = df.set_index(df.columns[0]).sum(axis=1)

plt.figure(figsize=(10,6))
total_by_size.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Total Deposits by Account Size")
plt.ylabel("Deposits")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("analysis_total_by_size.png")
plt.show()

# -------------------- Trend Analysis --------------------
if df.shape[1] > 2:
    df_melted = df.melt(id_vars=[df.columns[0]], var_name="Year/Quarter", value_name="Deposits")
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df_melted, x="Year/Quarter", y="Deposits", hue=df.columns[0], marker="o")
    plt.title("Trend of Deposits Over Time by Account Size")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("analysis_trend_over_time.png")
    plt.show()

# -------------------- Share & Growth --------------------
share = total_by_size / total_by_size.sum() * 100
print("\nShare of Each Account Size (%):\n", share)
print(f"Largest Category: {total_by_size.idxmax()}")
print(f"Smallest Category: {total_by_size.idxmin()}")

if df.shape[1] > 2:
    growth = df.set_index(df.columns[0]).pct_change(axis=1) * 100
    print("\nGrowth Rate (%):\n", growth)

cumulative = total_by_size.cumsum()
print("\nCumulative Deposits Distribution:\n", cumulative)

# -------------------- Correlation Heatmap --------------------
if df.shape[1] > 2:
    corr = df.iloc[:,1:].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap=sns.color_palette("YlOrRd", as_cmap=True))
    plt.title("Correlation Heatmap Between Years/Quarters")
    plt.tight_layout()
    plt.savefig("analysis_heatmap.png")
    plt.show()

# -------------------- Boxplot --------------------
if df.shape[1] > 2:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df.iloc[:,1:])
    plt.title("Boxplot of Deposits by Time Period")
    plt.ylabel("Deposits")
    plt.savefig("analysis_boxplot.png")
    plt.show()

# -------------------- Volatility & Outliers --------------------
std_dev = df.set_index(df.columns[0]).std(axis=1)
print("\nVolatility (Std Dev) per Account Size:\n", std_dev)

if df.shape[1] > 2:
    Q1 = df.iloc[:,1:].quantile(0.25)
    Q3 = df.iloc[:,1:].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df.iloc[:,1:] < (Q1 - 1.5 * IQR)) | (df.iloc[:,1:] > (Q3 + 1.5 * IQR)))
    print("\nOutliers Detected:\n", outliers.sum())

# -------------------- Pie Chart --------------------
plt.figure(figsize=(8,8))
total_by_size.plot(kind="pie", autopct="%1.1f%%", startangle=90, cmap="tab20")
plt.ylabel("")
plt.title("Share of Deposits by Account Size (Pie Chart)")
plt.tight_layout()
plt.savefig("analysis_pie_chart.png")
plt.show()

# -------------------- Survival Analysis --------------------
deposits = total_by_size.values
kmf = KaplanMeierFitter()
kmf.fit(deposits, event_observed=[1]*len(deposits))
plt.figure(figsize=(8,6))
kmf.plot()
plt.title("Survival Analysis of Deposits (Kaplan-Meier Curve)")
plt.xlabel("Deposit Amount")
plt.ylabel("Survival Probability")
plt.savefig("analysis_survival_curve.png")
plt.show()

# -------------------- Network Analysis --------------------


# -------------------- Monte Carlo Simulation --------------------
simulations = []
for _ in range(1000):
    growth_factors = np.random.normal(loc=1.05, scale=0.1, size=len(total_by_size))
    simulations.append((total_by_size.values * growth_factors).sum())

plt.figure(figsize=(8,6))
plt.hist(simulations, bins=30, color="purple", alpha=0.7)
plt.title("Monte Carlo Simulation: Future Total Deposits")
plt.xlabel("Simulated Total Deposits")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analysis_monte_carlo.png")
plt.show()

# -------------------- Bayesian Analysis --------------------
mean_deposit = np.mean(total_by_size.values)
a, b = 2, 2
posterior = beta(a + int(mean_deposit), b + len(total_by_size) - int(mean_deposit))
x = np.linspace(0, 1, 100)
plt.figure(figsize=(8,6))
plt.plot(x, posterior.pdf(x), label="Posterior")
plt.title("Bayesian Analysis: Posterior Distribution of Deposit Proportion")
plt.xlabel("Proportion")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("analysis_bayesian.png")
plt.show()

# -------------------- Scatterplot --------------------
if df.shape[1] > 2:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_melted, x="Year/Quarter", y="Deposits", hue=df.columns[0])
    plt.title("Scatterplot of Deposits by Account Size Over Time")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("analysis_scatterplot.png")
    plt.show()

# -------------------- KMeans Clustering --------------------
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:,1:].fillna(0))
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df["Cluster"] = clusters
print("\nMachine Learning Clustering Results:\n", df[[df.columns[0], "Cluster"]])

plt.figure(figsize=(8,6))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=clusters, palette="Set2", s=100)
plt.title("KMeans Clustering of Account Sizes")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.tight_layout()
plt.savefig("analysis_ml_clustering.png")
plt.show()
# -------------------- Network Analysis --------------------
#'''sizes = [str(s) for s in total_by_size.index.tolist()]
G = nx.Graph()
G.add_nodes_from(sizes)

# Add edges weighted by deposit differences
for i in range(len(sizes)):
    for j in range(i+1, len(sizes)):
        weight = abs(total_by_size.iloc[i] - total_by_size.iloc[j])
        G.add_edge(sizes[i], sizes[j], weight=weight)

# NetworkX 2.5 compatible layout (no seed/random_state)
pos = nx.spring_layout(G, k=0.1, iterations=50)

plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

plt.title("Network Analysis of Account Sizes (Deposit Differences)")
plt.tight_layout()
plt.savefig("analysis_network.png")
plt.show()