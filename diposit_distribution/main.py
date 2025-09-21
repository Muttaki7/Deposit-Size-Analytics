print("libs")
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import networkx as nx
from itertools import combinations
import warnings
from scipy.stats import zscore
print("------------------------------")
warnings.filterwarnings('ignore')

csv_file = r"C:\Users\Muttaki\Desktop\analysis gov\data-resource_2024_06_24_Table-18  Deposits  distributed by Size of Account.csv"

df = pd.read_csv(csv_file, skiprows=5, header=0)

num_periods = 28
col_names = ['Account_Size']
for i in range(num_periods):
    year = 2018 + (i // 4)
    quarter = (i % 4) + 1
    col_names.extend([f'{year}_Q{quarter}_Accounts', f'{year}_Q{quarter}_Amount'])
df.columns = col_names[:len(df.columns)]
df = df[df['Account_Size'].str.contains('Note|Source|Table', na=False) == False]
numeric_cols = df.columns[1:]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(how='all')
account_sizes = df['Account_Size'].tolist()
print("Account Sizes:", account_sizes)
years = range(2018, 2025)
annual_accounts = []
annual_deposits = []
for year in years:
    acc_sum = 0
    dep_sum = 0
    for q in range(1, 5):
        if year == 2024 and q > 1:
            break
        col_acc = f'{year}_Q{q}_Accounts'
        col_dep = f'{year}_Q{q}_Amount'
        if col_acc in df.columns:
            acc_sum += df[col_acc].sum()
            dep_sum += df[col_dep].sum()
    annual_accounts.append(acc_sum)
    annual_deposits.append(dep_sum)

trend_df = pd.DataFrame({'Year': years, 'Total_Accounts': annual_accounts, 'Total_Deposits': annual_deposits})
print("Annual Trend",trend_df)
print("Plots")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(trend_df['Year'], trend_df['Total_Accounts']/1e6)  # in millions
plt.title('Accounts Trend')
plt.ylabel('Millions')
plt.subplot(1, 2, 2)
plt.plot(trend_df['Year'], trend_df['Total_Deposits']/1e3)  # in thousands crore
plt.title('Deposits Trend')
plt.ylabel('Thousand Crore Tk')
plt.tight_layout()
plt.savefig('trend.png')
plt.show()

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
q_deposits = {q: [] for q in quarters}

for year in years:
    for q in quarters:
        col = f'{year}_{q}_Amount'
        if col in df.columns:
            q_deposits[q].append(df[col].sum())
        else:
            q_deposits[q].append(np.nan)   # pad with NaN if missing

q_df = pd.DataFrame(q_deposits, index=years).T
print("Quarterly Deposits Comparison:", q_df)
print("plot")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)

sns.heatmap(q_df, annot=True, cmap='YlOrRd')
plt.title('Quarterly Deposits Heatmap')
plt.savefig('quarterly_heatmap.png')
plt.show()
avg_dep_per_size = df.iloc[:-1,2::2].mean(axis=1)
dominance = pd.DataFrame({'Size':account_sizes[:-1],'Avg_deopsit':avg_dep_per_size.values})
dominance = dominance.sort_values(by='Avg_deopsit',ascending=False)
print("dominenting Accounts:",dominance.head())
print("Pie Chart")
latest_year = 2023
latest_dep = df[[f'{latest_year}_Q{q}_Amount' for q in range(1, 5)]].sum(axis=1).iloc[:-1]
plt.pie(latest_dep,labels=account_sizes[:-1],autopct='%1.2f%%')
plt.title('Latest Deposits Comparison {latest_year}')
plt.savefig('latest_deposits_comparison.png')
plt.show()
print("Groth Rate")
trend_df['Dep_Growth'] = trend_df['Total_Deposits'].pct_change() * 100
trend_df['Acc_Growth'] = trend_df['Total_Accounts'].pct_change() * 100
print("\n4. Annual Growth Rates:")
print(trend_df[['Year', 'Dep_Growth', 'Acc_Growth']])
total_q_dep = []
quarters_full = []
for year in years:
    for q in range(1,5):
        if year == 2024 and q > 1:
            break
        col = f'{year}_Q{q}_Amount'
        if col in df.columns:
            total_q_dep.append(df[col].sum())
            quarters_full.append(f'{year} Q{q}')

q_growth = pd.Series(total_q_dep).pct_change() * 100
q_growth_df = pd.DataFrame({'Quarter': quarters_full[1:], 'Q_Growth': q_growth[1:]})
print("annual Groth Rares Sample",q_growth_df.tail())
print("Smallest And Largest Contributions")
smallest = df.iloc[0, 2::2].sum() / trend_df['Total_Deposits'].sum() * 100
largest = df.iloc[-2, 2::2].sum() / trend_df['Total_Deposits'].sum() * 100
print(f"\n5. Smallest Accounts Contribution: {smallest:.2f}% over period")
print(f"Largest Accounts Contribution: {largest:.2f}% over period")

plt.bar(['Smallest (Upto 5k)', 'Largest (50M+)'], [smallest, largest])
plt.title('Top vs Bottom Contribution')
plt.ylabel('% of Total Deposits')
plt.savefig('top_bottom.png')
plt.show()
print("Comulative Deposits sum deposit over time ")
cum_dep = np.cumsum(total_q_dep)
cum_df = pd.DataFrame({'Quarter': quarters_full, 'Cum_Deposits': cum_dep})
print("\n6. Cumulative Deposits (last 5):")
print(cum_df.tail())

plt.plot(cum_df['Quarter'], cum_df['Cum_Deposits']/1e3)
plt.title('Cumulative Deposits')
plt.xticks(rotation=45)
plt.ylabel('Thousand Crore Tk')
plt.savefig('cumulative.png')
plt.show()
print("Average Deposits per Account")
avg_per_size = []
for i in range(len(account_sizes)-1):
    acc_means = df.iloc[i, 1::2].mean()
    dep_means = df.iloc[i, 2::2].mean()
    avg_dep_acc = dep_means / acc_means if acc_means > 0 else 0
    avg_per_size.append(avg_dep_acc)

avg_df = pd.DataFrame({'Size': account_sizes[:-1], 'Avg_Deposit_per_Account': avg_per_size})
print("\n7. Average Deposit per Account per Size (sample):")
print(avg_df.head())

plt.barh(avg_df['Size'][:10], avg_df['Avg_Deposit_per_Account'][:10])
plt.title('Avg Deposit per Account by Size')
plt.savefig('avg_per_acc.png')
plt.show()
print("Volatility Analysis")
q_vol = pd.Series(total_q_dep).rolling(10).std
print("Quaterly Volatility Analysis based on stddev and sample:",pd.DataFrame({'Quarter': quarters_full, 'Volatility': q_vol}).tail())
print("Percentage share calculation of each account")
latest_col = f'2024_Q1_Amount'
if latest_col not in df.columns:
    latest_col = df.filter(like='_Amount').columns[-1]
latest_total = df[latest_col].sum()
contrib_pct = (df[latest_col].iloc[:-1] / latest_total * 100).round(2)
contrib_df = pd.DataFrame({'Size': account_sizes[:-1], 'Contrib_%': contrib_pct})
print("Contribution Percentage per Account:",contrib_df.sort_values('Contrib_%', ascending=False).head())
print("Forcasting Using ARIMA and HOLT-Winter ")
ts_dep = pd.Series(total_q_dep, index=pd.date_range(start='2018-03-31', periods=len(total_q_dep), freq='3M'))
model_arima = ARIMA(ts_dep, order=(1,1,1))
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=5)
print("ARIMA Forecast:",forecast_arima)
model_hw = ExponentialSmoothing(ts_dep, seasonal='add', seasonal_periods=4)
fit_hw = model_hw.fit()
forecast_hw = fit_hw.forecast(steps=5)
print("Holt Winter Forecast:",forecast_hw)
plt.plot(ts_dep, label='Historical')
plt.plot(pd.date_range(start=ts_dep.index[-1] + pd.DateOffset(months=3), periods=5, freq='3M'), forecast_arima, label='ARIMA')
plt.plot(pd.date_range(start=ts_dep.index[-1] + pd.DateOffset(months=3), periods=5, freq='3M'), forecast_hw, label='HW')
plt.legend()
plt.title('Deposit Forecast')
plt.savefig('forecast.png')
plt.show()
print("Yearly Inensity Heatmap")
yearly_heatmap = trend_df.set_index('Year')['Total_Deposits'].to_frame().T
sns.heatmap(yearly_heatmap, annot=True, cmap='YlOrRd')
plt.title('Yearly Deposits Heatmap')
plt.savefig('yearly_heatmap.png')
plt.show()
print("Clustering Group Account Size")
features = []
for i in range(len(account_sizes)-1):
    acc_avg = df.iloc[i, 1::2].mean()
    dep_avg = df.iloc[i, 2::2].mean()
    features.append([acc_avg, dep_avg])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

cluster_df = pd.DataFrame({'Size': account_sizes[:-1], 'Cluster': clusters, 'Avg_Acc': [f[0] for f in features], 'Avg_Dep': [f[1] for f in features]})
print("\n12. Clustering Results:")
print(cluster_df.groupby('Cluster')[['Avg_Acc', 'Avg_Dep']].mean())


pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)
plt.scatter(features_pca[:,0], features_pca[:,1], c=clusters)
for i, size in enumerate(account_sizes[:-1]):
    plt.annotate(size[:20], (features_pca[i,0], features_pca[i,1]))
plt.title('Clustering of Account Sizes')
plt.savefig('clustering.png')
plt.show()
print("Corelation Between Accounts And Deposits")
corr = stats.pearsonr(annual_accounts, annual_deposits)
print(f"\n13. Correlation Accounts vs Deposits: {corr[0]:.2f} (p={corr[1]:.4f})")

q_acc = []
for year in years:
    for q in range(1,5):
        if year == 2024 and q > 1: break
        col = f'{year}_Q{q}_Accounts'
        if col in df.columns:
            q_acc.append(df[col].sum())
corr_q = stats.pearsonr(q_acc, total_q_dep)
print(f"Quarterly Correlation: {corr_q[0]:.2f}")
print("Survival Analysis Using Retention ")
retention = []
for i in range(len(account_sizes)-1):
    deps = df.iloc[i, 2::2].values
    increasing = np.sum(np.diff(deps) > 0) / (len(deps)-1) * 100
    retention.append(increasing)

ret_df = pd.DataFrame({'Size': account_sizes[:-1], 'Retention_%': retention})
print("Retention Rate (% increasing quarters):")
print(ret_df.sort_values('Retention_%', ascending=False).head())
print("Simulation:")
n_sims = 1000
future_quarters = 4
sims = np.zeros((n_sims, future_quarters))
current_dep = total_q_dep[-1]
growth_mean = 0.05 / 4
growth_std = 0.02 / 4
for i in range(n_sims):
    growths = np.random.normal(growth_mean, growth_std, future_quarters)
    sims[i] = current_dep * np.cumprod(1 + growths)

mc_mean = np.mean(sims, axis=0)
mc_ci = np.percentile(sims, [5, 95], axis=0)
print("\n15. Monte Carlo Simulation (Mean next 4 quarters):")
for i, m in enumerate(mc_mean):
    print(f"Q{i+1}: {m:.2f} (95% CI: {mc_ci[0,i]:.2f} - {mc_ci[1,i]:.2f})")

plt.plot(mc_mean, label='Mean')
plt.fill_between(range(future_quarters), mc_ci[0], mc_ci[1], alpha=0.3)
plt.title('MC Simulation')
plt.savefig('mc_sim.png')
plt.show()
print("Network Analysis")
G = nx.Graph()
G.add_nodes_from(years, bipartite=0)
G.add_nodes_from(account_sizes[:-1], bipartite=1)
for year in years:
    for i, size in enumerate(account_sizes[:-1]):
        dep_cols = [f'{year}_Q{q}_Amount' for q in range(1,5) if f'{year}_Q{q}_Amount' in df.columns]
        if dep_cols:
            avg_dep_year = df[dep_cols].iloc[i].mean()
            if not np.isnan(avg_dep_year) and avg_dep_year > np.median(df[dep_cols].mean()):
                G.add_edge(year, size, weight=avg_dep_year)

pos = nx.bipartite_layout(G, years)
nx.draw(G, pos, with_labels=True)
plt.title('Network: Years-Account Sizes')
plt.savefig('network.png')
plt.show()
print(f"Network: {len(G.edges)} relations built")
print("bAYSIAN Analysis")
data_growth = trend_df['Dep_Growth'].dropna().values
prior_mean, prior_std = 5, 3
posterior = norm(loc=np.mean(data_growth), scale=np.std(data_growth)/np.sqrt(len(data_growth)))
print("Bayesian: Posterior mean growth {:.2f}%, std {:.2f}%".format(posterior.mean(), posterior.std()))
print("bar and Pie Chart")
plt.barh(dominance['Size'][:5], dominance['Avg_Deposit'][:5])
plt.title('Top 5 Account Sizes by Avg Deposit')
plt.savefig('bar_dist.png')
plt.show()
print("Ranking")
max_year = trend_df.loc[trend_df['Total_Deposits'].idxmax(), 'Year']
max_q = max(enumerate(total_q_dep), key=lambda x: x[1])
max_q_label = quarters_full[max_q[0]]
print("Highest Year: {max_year}, Highest Quarter: {max_q_label}")
print("Outlier Detection")
z_scores = np.abs(zscore(total_q_dep))
outliers = [quarters_full[i] for i in np.where(z_scores > 2)[0]]
print(f"\nOutlier Quarters (z>2): {outliers}")

Q1 = np.percentile(total_q_dep, 25)
Q3 = np.percentile(total_q_dep, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
iqr_out = [quarters_full[i] for i, v in enumerate(total_q_dep) if v < lower or v > upper]
print(f"IQR Outliers: {iqr_out}")
print("Analysis complete. Plots saved as PNG files.")