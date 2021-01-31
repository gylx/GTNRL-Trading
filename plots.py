import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


def get_testing_df (agent='GTN', ep='14'):

    # Returns averaged across models per episodes plotting
    files = glob.glob('results/**.csv')
    files = [file for file in files if ('testing' in file) and (agent in file)]
    col_names = [(n.split(f'{agent} ')[-1]).split(' testing')[0] for n in files]

    testing_df = [pd.read_csv(file, index_col=0, header=0)[ep] for file in files if 'testing' in file]
    testing_df = pd.concat(testing_df, axis=1)
    testing_df.columns = col_names

    # european currencies only
    sel = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDNOK', 'USDSEK']
    testing_df = testing_df[sel]

    return testing_df


def compute_metrics (rs):

    # Total return
    total_r = np.exp(rs.sum()*0.01)-1

    # Sharpe ratio
    sharpe = rs.mean()/rs.std()

    # Sortino ratio
    sortino = rs.mean()/rs[rs<0].std()

    # Maximum Draw-Down
    current_sum = 0
    max_sum = 0
    for n in -rs:
        current_sum = max(0, current_sum + n)
        max_sum = max(current_sum, max_sum)
    max_dd = np.exp(max_sum*0.01)-1

    # Hit ratio
    hit_ratio = rs.apply(np.sign).replace(-1,0).mean()

    # Results
    results = pd.Series(data=[total_r, sharpe, sortino, max_dd, hit_ratio],
                        index=['Total Return', 'Sharpe', 'Sortino', 'Max DD', 'Hit Rate'])

    return results


agents = ['GTN', 'RNN', 'TTNN', 'GNN']
all_rs = []
for agent in agents:
    all_rs.append(get_testing_df(agent=agent).mean(1).copy())

plt.rcParams['font.size'] = 20
all_cumrs = 1000*((0.01*pd.concat(all_rs, axis=1)).cumsum().apply(np.exp))
all_cumrs.plot(figsize=(12, 5), linewidth=3, grid=True)
plt.legend([a.replace('RNN', 'GRU').replace('GNN', 'GCN').replace('GTN', 'fMGTN') for a in agents])
plt.title('Test-Set Trading Performance')
# plt.xlabel('minutes')
# plt.ylabel('portfolio value')
plt.tight_layout()
plt.savefig('backtest-performance.png')

results = pd.concat([compute_metrics(rs) for rs in all_rs], axis=1)
results.columns = [a.replace('RNN', 'GRU').replace('GNN', 'GCN').replace('GTN', 'fMGTN') for a in agents]
print(results.T)