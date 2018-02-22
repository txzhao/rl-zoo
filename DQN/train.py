import os, sys
dir = os.path.dirname(__file__)
pathname = os.path.join(dir, '../')
sys.path.append(os.path.realpath(pathname))

import time
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import gym
from dqn import DQN
import utils


# Hyper Parameters
MaxEpisodes = 300
num_of_runs = 5
winWidth = 100
writeCSV = True
savePlot = True

env_id = 'CartPole-v0'
config_file = 'ddqn_cartpole.json'
env = gym.make(env_id)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
env_config = {'n_actions': N_ACTIONS, 'n_states': N_STATES, 'env_a_shape': ENV_A_SHAPE}


if __name__ == '__main__':
	aver_rwd_dqn = np.array((MaxEpisodes, ))

	for exp in range(num_of_runs):
		print('\nExperiment NO.' + str(exp+1))

		# agent spec
		config = json.load(open('./configs/' + config_file))
		config.update(env_config)
		dqn_agent = DQN(config)

		EPSILON = config['exploration']['init_epsilon']
		total_steps = 0
		rwd_dqn = []

		for i_episode in range(MaxEpisodes):
			s = env.reset()
			ep_r = 0
			timestep = 0
			while True:
				total_steps += 1

				# decay exploration
				EPSILON = utils.epsilon_decay(
					eps=EPSILON, 
					step=total_steps, 
					config=config['exploration']
				)

				# env.render()
				a = dqn_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				if config['modify_reward']:
					r = utils.modify_rwd(env_id, s_)

				# store current transition
				dqn_agent.store_transition(s, a, r, s_, done)
				timestep += 1

				# start update policy when memory has enough exps
				if dqn_agent.memory_counter > config['first_update']:
					dqn_agent.learn()

				if done:
					prefix = 'DDQN' if config['double_q_model'] else 'DQN'
					if config['memory']['prioritized']: prefix += '-PER'
					print(prefix + ' - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_dqn.append(ep_r)
					break
				s = s_

		del dqn_agent

		# incrementally calculate mean and variance
		rwd_dqn = utils.moving_avg(rwd_dqn, winWidth)
		tmp_rwd = np.array(rwd_dqn)
		pre_rwd = aver_rwd_dqn
		aver_rwd_dqn = aver_rwd_dqn + (tmp_rwd - aver_rwd_dqn)/float(exp+1)
		if exp == 0:
			var_rwd = np.zeros(aver_rwd_dqn.shape)
		else:
			var_rwd = var_rwd + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_dqn))/float(exp+1)
			var_rwd = var_rwd/float(exp+1)

	env.close()

	# save data to csv
	if writeCSV:
		data = {'episodes': range(MaxEpisodes), 'rewards': list(aver_rwd_dqn), 'variances': list(np.sqrt(var_rwd))}
		df = pd.DataFrame(data=dict([(key, pd.Series(value)) for key, value in data.items()]),
			index=range(0, MaxEpisodes),
			columns=['episodes', 'rewards', 'variances'])
		if config['double_q_model']:
			if config['memory']['prioritized']:
				df.to_csv('../results/logs/ddqn_per_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
			else:
				df.to_csv('../results/logs/ddqn_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
		else:
			if config['memory']['prioritized']:
				df.to_csv('../results/logs/dqn_per_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
			else:
				df.to_csv('../results/logs/dqn_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))

	# Save reward plot
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_dqn), 'k', label='no pre-training')
	ax.fill_between(range(MaxEpisodes), aver_rwd_dqn + np.sqrt(var_rwd), aver_rwd_dqn - np.sqrt(var_rwd), facecolor='black', alpha=0.1)
	ax.set_title('Number of Run: ' + str(num_of_runs))
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	ax.legend(loc='upper left')
	ax.grid()
	if config['double_q_model']:
		if config['memory']['prioritized']:
			fig.savefig('../results/figures/ddqn_per_{}.png'.format(int(time.time())))
		else:
			fig.savefig('../results/figures/ddqn_{}.png'.format(int(time.time())))
	else:
		if config['memory']['prioritized']:
			fig.savefig('../results/figures/dqn_per_{}.png'.format(int(time.time())))
		else:
			fig.savefig('../results/figures/dqn_{}.png'.format(int(time.time())))
	plt.close(fig)

