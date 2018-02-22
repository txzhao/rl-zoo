import time
import csv
import numpy as np
from matplotlib import pyplot as plt

num_of_runs = 5
steps = 300
files = ['dqn_steps_300_run_5.csv', 'ddqn_steps_300_run_5.csv', 'dqn_per_steps_300_run_5.csv', 'ddqn_per_steps_300_run_5.csv']
labels = ['DQN', 'DDQN', 'DQN-PER', 'DDQN-PER']
colors = ['b', 'r', 'k', 'm']
epList, rwdList, varList = [], [], []

for filename in files:
	episodes, rewards, variances = [], [], []
	with open(filename) as csvfile:
		next(csvfile)
		datareader = csv.reader(csvfile, delimiter=',')
		for row in datareader:
			episodes.append(float(row[1]))
			rewards.append(float(row[2]))
			variances.append(float(row[3]))

	epList.append(episodes)
	rwdList.append(rewards)
	varList.append(variances)


fig, ax = plt.subplots(nrows=1, ncols=1)
for i in range(len(files)):
	episodes = epList[i]
	rewards = np.array(rwdList[i])
	variances = np.array(varList[i])
	ax.plot(episodes, list(rewards), colors[i], label=labels[i])
	ax.fill_between(episodes, list(rewards + np.sqrt(variances)), list(rewards - np.sqrt(variances)), facecolor=colors[i], alpha=0.1)

ax.set_xlabel('Episodes')
ax.set_ylabel('Average Rewards')
ax.legend(loc='upper left')
ax.grid()
fig.savefig('../figures/trial_{}.png'.format(int(time.time())))
plt.close(fig)
