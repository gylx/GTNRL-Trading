# Import relevant modules
from agent.agent_DDQNN import RNNAgent, GTNAgent, TTNNAgent, GNNAgent
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

# Set random seed
random.seed(0)
np.random.seed(0)

# Initialize Agent variables
trading_currency = 'USDSEK'
window_size = 30
episode_count = 15
batch_size = 64  # batch size for replaying/training the agent
agent_type = 'GNN'  # RNN or GTN or TTNN or GNN

# Initialize training variables
total_rewards_df = pd.DataFrame(dtype=float)

# Get returns data
rs_types = ['open', 'high', 'low', 'last']
file_names = [f'g10_minute_{t}_rs_2019-10-01.csv' for t in rs_types]
rs_data = dict(zip(rs_types, [pd.read_csv(f'data/{f}', index_col=0, header=0) for f in file_names]))
rs_y = rs_data['last'][trading_currency]

# Get graphs data
A_t = pd.read_csv('data/A_t_22.csv', index_col=0, header=0)
A_n = pd.read_csv('data/g10_daily_carry_adjacency_matrix_2000_2019.csv', index_col=0, header=0)
graph_list = [A_t.values, A_n.values]

# Training vs Evaluation
split = int(0.7*rs_y.shape[0])
rs_y = rs_y.iloc[split:]

# RNN Agent Setup
if agent_type == 'RNN':
    state_size = (window_size, rs_data['last'].shape[1])
    rs_data = pd.concat(list(rs_data.values()), 1)

# Evaluate over episodes

'''
TODO:
- Iterate over models for each of the pisodes
- For each model, run it over the entire validation period
- Plot and save plot
'''

for e in range(episode_count):

    # Load agent
    if agent_type == 'RNN':
        agent = RNNAgent(state_size=state_size,
                         model_name=f'model_ep{e}',
                         model_target_name=f'model_target_ep{e}',
                         is_eval=True)

    elif agent_type == 'GTN':
        agent = GTNAgent(state_size=(window_size, rs_data['last'].shape[1], len(rs_types)),
                         graph_list=graph_list,
                         model_name=f'model_ep{e}',
                         model_target_name=f'model_target_ep{e}',
                         is_eval=True)

    elif agent_type == 'TTNN':
        agent = TTNNAgent(state_size=(window_size, rs_data['last'].shape[1], len(rs_types)),
                          model_name=f'model_ep{e}',
                          model_target_name=f'model_target_ep{e}',
                          is_eval=True)

    elif agent_type == 'GNN':
        agent = GNNAgent(state_size=(A_n.shape[0], len(rs_types) * window_size),
                         graph_adj=A_n.values,
                         model_name=f'model_ep{e}',
                         model_target_name=f'model_target_ep{e}')

    # Print progress
    print(f"Episode: {e + 1}/{episode_count}")
    print(f"Epsilon: {agent.epsilon}")

    # Reset agent parameters to run next episode
    agent.episode_reset()

    # Loop over time
    for t in rs_y.index[window_size:]:

        # past {window_size} log returns up to and excluding {t}
        if agent_type == 'RNN':
            X = rs_data.loc[:t].iloc[-window_size-1:-1]  # fetch raw data
            X = X.values.reshape([1]+list(X.shape))  # tensorize
        elif (agent_type == 'GTN') or (agent_type == 'TTNN'):
            X = np.array([rs_data[k].loc[:t].iloc[-window_size - 1:-1].values for k in rs_data.keys()])
            X = X.transpose([1, 2, 0])
            X = X.reshape([1] + list(X.shape))
        elif agent_type == 'GNN':
            X = np.array([rs_data[k].loc[:t].iloc[-window_size-1:-1].values for k in rs_data.keys()])
            X = X.transpose([2,1,0])
            X = X.reshape(-1, np.prod(X.shape[1:]))
            X = X.reshape([1]+list(X.shape))

        # Get action from agent
        action = agent.act(X)

        # Process returns/rewards
        action_direction = -1 * (action * 2 - 1)  # map 0->buy->+1, 1->sell->-1
        reward = 100 * action_direction * rs_y[t]
        agent.episode_tot_reward += reward
        agent.episode_rewards.append(reward)
        print(t, agent.episode_tot_reward, reward)

        # Fetch next state
        done = True if t == rs_y.index[-1] else False
        if agent_type == 'RNN':
            next_X = rs_data.loc[:t].iloc[-window_size:]  # fetch raw data
            next_X = next_X.values.reshape([1]+list(next_X.shape))  # tensorize
        elif (agent_type == 'GTN') or (agent_type == 'TTNN'):
            next_X = np.array([rs_data[k].loc[:t].iloc[-window_size:].values for k in rs_data.keys()])
            next_X = next_X.transpose([1, 2, 0])
            next_X = next_X.reshape([1] + list(next_X.shape))
        elif agent_type == 'GNN':
            next_X = np.array([rs_data[k].loc[:t].iloc[-window_size:].values for k in rs_data.keys()])
            next_X = next_X.transpose([2,1,0])
            next_X = next_X.reshape(-1, np.prod(next_X.shape[1:]))
            next_X = next_X.reshape([1]+list(next_X.shape))

        # Append to memory & train
        # agent.memory.append((X[0], action, reward, next_X[0], done))
        # agent.replay(min(batch_size, len(agent.memory)))

        # Print if done
        if done:
            print("--------------------------------")
            print(f"Episode reward:{agent.episode_tot_reward}%")
            print("--------------------------------")

    # Record episode data
    total_rewards_df[e] = agent.episode_rewards

plt.figure()
total_rewards_df.cumsum().plot()
plt.savefig(f'results/{agent_type} (EVALUATION) strategy returns.png')

plt.figure()
total_rewards_df.sum().plot()
plt.savefig(f'results/{agent_type} (EVALUATION) total returns across episodes.png')

total_rewards_df.to_csv(f'results/{agent_type} {trading_currency} testing rewards df.csv')