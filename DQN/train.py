import os, sys
dir = os.path.dirname(__file__)
pathname = os.path.join(dir, '../')
sys.path.append(os.path.realpath(pathname))

import time
import numpy as np
from matplotlib import pyplot as plt
import gym
from dqn import DQN
import utils


# Hyper Parameters
MaxEpisodes = 400
num_of_runs = 3
winWidth = 100

env_id = 'CartPole-v0'
env = gym.make(env_id)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


if __name__ == '__main__':
	aver_rwd_epc = np.array((MaxEpisodes, ))

	for exp in range(num_of_runs):
		print('\nExperiment NO.' + str(exp+1))

		# agent spec
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.8,
			'min_epsilon': 0.01,
			'decay_steps': 10000,
			'decay_eps': 0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 50000,
			'first_update': 1000
		}
		agent = DQN(config)

		EPSILON = config['init_epsilon']
		rwd_epc = []

		for i_episode in range(MaxEpisodes):
			s = env.reset()
			ep_r = 0
			timestep = 0
			while True:
				# decay exploration
				EPSILON = utils.epsilon_decay_exp(eps=EPSILON, min_eps=config['min_epsilon'], decay=config['decay_eps'])

				# env.render()
				a = agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				# store current transition
				agent.store_transition(s, a, r, s_, done)
				timestep += 1

				# start update policy when memory has enough exps
				if agent.memory_counter > config['first_update']:
					agent.learn()

				if done:
					print('Pretrain - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_epc.append(ep_r)
					break
				s = s_

		del agent

		# incrementally calculate mean and variance
		rwd_epc = utils.moving_avg(rwd_epc, winWidth)
		tmp_rwd = np.array(rwd_epc)
		pre_rwd = aver_rwd_epc
		aver_rwd_epc = aver_rwd_epc + (tmp_rwd - aver_rwd_epc)/float(exp+1)
		if exp == 0:
			var_rwd = np.zeros(aver_rwd_epc.shape)
		else:
			var_rwd = var_rwd + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_epc))/float(exp+1)
			var_rwd = var_rwd/float(exp+1)

	env.close()

	# Save reward plot
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_epc), 'k', label='no pre-training')
	ax.fill_between(range(MaxEpisodes), aver_rwd_epc + np.sqrt(var_rwd), aver_rwd_epc - np.sqrt(var_rwd), facecolor='black', alpha=0.1)
	ax.set_title('Number of Run: ' + str(num_of_runs))
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	ax.legend(loc='upper left')
	ax.grid()
	fig.savefig('./Figure/result_{}.png'.format(int(time.time())))
	plt.close(fig)

