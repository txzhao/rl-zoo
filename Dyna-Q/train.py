import time
import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import gym
from dyna_q import DynaQ
import utils

# Hyper Parameters
Pretrain_steps = 200
GAN_pretrain_steps = 600
Epochs = 320
MaxEpisodes = 300
GANSampleSzie = 30000
num_of_runs = 5
print_per_epoch = 5
winWidth = 100
K = 5

env_id = 'MountainCar-v0'
env = gym.make(env_id)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


if __name__ == '__main__':
	aver_rwd_dyna = np.array((MaxEpisodes, ))
	for exp in range(num_of_runs):
		rwd_dyna = []
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.8,
			'min_epsilon': 1e-4,
			'decay_steps': 10000,
			'decay_eps': 0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 20000,
			'first_update': 1000
		}
		dyna_q_agent = DynaQ(config)

		EPSILON = config['init_epsilon']
		delta_eps = (config['init_epsilon'] - config['min_epsilon'])/float(config['decay_steps'])

		print('\nDyna-Q agent...')
		for i_episode in range(MaxEpisodes):
			s = env.reset()
			ep_r = 0
			timestep = 0
			while True:
				# decay exploration
				EPSILON = utils.epsilon_decay_exp(eps=EPSILON, min_eps=config['min_epsilon'], decay=config['decay_eps'])
				# EPSILON = utils.epsilon_linear_anneal(
				# 	eps=EPSILON, ini_eps=config['init_epsilon'], min_eps=config['min_epsilon'], timesteps=config['decay_steps'])

				#env.render()
				a = dyna_q_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				dyna_q_agent.store_transition(s, a, r, s_, done)
				timestep += 1
				if dyna_q_agent.memory_counter > config['first_update']:
					dyna_q_agent.learn()
					dyna_q_agent.update_env_model()

				for _ in range(K):
					dyna_q_agent.simulate_learn()

				if done:
					print('Dyna-Q - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_dyna.append(ep_r)
					break
				s = s_

		rwd_dyna = utils.moving_avg(rwd_dyna, winWidth)
		tmp_rwd = np.array(rwd_dyna)
		aver_rwd_dyna = aver_rwd_dyna + (tmp_rwd - aver_rwd_dyna)/float(exp+1)
		if exp == 0:
			var_rwd_dyna = np.zeros(aver_rwd_dyna.shape)
		else:
			var_rwd_dyna = float(exp-1)/float(exp)*var_rwd_dyna + np.square(tmp_rwd - aver_rwd_dyna)/float(exp+1)

	env.close()

	# Save reward plot
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_dyna), 'k', label='Dyna-Q')
	ax.fill_between(range(MaxEpisodes), aver_rwd_dyna + np.sqrt(var_rwd_dyna), aver_rwd_dyna - np.sqrt(var_rwd_dyna), facecolor='black', alpha=0.2)
	ax.set_title('Number of Run: ' + str(num_of_runs))
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	ax.legend(loc='upper left')
	ax.grid()
	fig.savefig('./Figure/result_{}.png'.format(int(time.time())))
	plt.close(fig)

