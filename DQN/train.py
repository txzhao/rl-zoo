import time
import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import gym
from dqn import DQN
from gan import GAN
import utils


# Hyper Parameters
Pretrain_steps = 400
GAN_pretrain_steps = 800
Epochs = 320
MaxEpisodes = 400
GANSampleSzie = 30000
num_of_runs = 3
winWidth = 100
loss_verbose = False

env_id = 'CartPole-v0'
env = gym.make(env_id)
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


if __name__ == '__main__':
	aver_rwd_no_pretrain = aver_rwd_pretrain = aver_rwd_only_pretrain = aver_rwd_set_memory = aver_rwd_no_pretrain_dqn = np.array((MaxEpisodes, ))

	for exp in range(num_of_runs):
		print('\nExperiment NO.' + str(exp+1))
		rwd_no_pretrain, rwd_pretrain, rwd_only_pretrain, rwd_set_memory, rwd_no_pretrain_dqn = [], [], [], [], []

		# ======================== experience generator ========================
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
		exp_generator = DQN(config)
		experiences = []
		xsList, ysList, rsList, dsList = [], [], [], []

		EPSILON = config['init_epsilon']
		startSample = 100
		EndSample = 200

		print('\nCollecting experience...')
		for i_episode in range(EndSample):
			s = env.reset()
			ep_r = 0
			timestep = 0
			while True:
				# decay exploration
				EPSILON = utils.epsilon_decay_exp(eps=EPSILON, min_eps=config['min_epsilon'], decay=config['decay_eps'])
				# EPSILON = utils.epsilon_linear_anneal(
				# 	eps=EPSILON, ini_eps=config['init_epsilon'], min_eps=config['min_epsilon'], timesteps=config['decay_steps'])

				#env.render()
				a = exp_generator.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				exp_generator.store_transition(s, a, r, s_, done)
				timestep += 1
				if exp_generator.memory_counter > config['first_update']:
					exp_generator.learn()

				if i_episode >= startSample - 1:
					experience = np.hstack((s, [a, r, done], s_))
					experiences.append(experience)
					xsList.append(s)
					ysList.append(a)
					rsList.append(r)
					dsList.append(done)

				if done:
					# print('Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					break
				s = s_

		del exp_generator
		print('\nCollecting experience finished!')


	# ======================== no pretrain ========================
		print('\nNo-pretrain agent...')
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.01,
			'min_epsilon': 0.01,
			'decay_steps': 10000,
			'decay_eps':0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 50000,
			'first_update': 2000
		}
		no_pretrain_agent = DQN(config)

		EPSILON = config['init_epsilon']

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
				a = no_pretrain_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				no_pretrain_agent.store_transition(s, a, r, s_, done)
				timestep += 1
				if no_pretrain_agent.memory_counter > config['first_update']:
					no_pretrain_agent.learn()
				if done:
					print('No pretrain - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_no_pretrain.append(ep_r)
					break

				s = s_

		del no_pretrain_agent

		rwd_no_pretrain = utils.moving_avg(rwd_no_pretrain, winWidth)
		tmp_rwd = np.array(rwd_no_pretrain)
		pre_rwd = aver_rwd_no_pretrain
		aver_rwd_no_pretrain = aver_rwd_no_pretrain + (tmp_rwd - aver_rwd_no_pretrain)/float(exp+1)
		if exp == 0:
			var_rwd_no_pretrain = np.zeros(aver_rwd_no_pretrain.shape)
		else:
			# var_rwd_no_pretrain = float(exp-1)/float(exp)*var_rwd_no_pretrain + np.square(tmp_rwd - aver_rwd_no_pretrain)/float(exp+1)
			var_rwd_no_pretrain = var_rwd_no_pretrain + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_no_pretrain))/float(exp+1)
			var_rwd_no_pretrain = var_rwd_no_pretrain/float(exp+1)


		# config = {
		# 	'n_actions': N_ACTIONS,
		# 	'n_states': N_STATES,
		# 	'env_a_shape': ENV_A_SHAPE,
		# 	'double_q_model': False,
		# 	'batch_size': 128,
		# 	'learning_rate': 5e-3,
		# 	'init_epsilon': 0.8,
		# 	'min_epsilon': 0.01,
		# 	'decay_steps': 10000,
		# 	'decay_eps':0.99,
		# 	'discount': 0.99,
		# 	'target_update_freq': 500,
		# 	'memory_capacity': 50000,
		# 	'first_update': 1000
		# }
		# no_pretrain_agent_1 = DQN(config)

		# EPSILON = config['init_epsilon']
		# delta_eps = (config['init_epsilon'] - config['min_epsilon'])/float(config['decay_steps'])

		# for i_episode in range(MaxEpisodes):
		# 	s = env.reset()
		# 	ep_r = 0
		# 	timestep = 0
		# 	while True:
		# 		# decay exploration
		# 		EPSILON *= config['decay_eps']
		# 		# EPSILON -= delta_eps
		# 		EPSILON = max(EPSILON, config['min_epsilon'])

		# 		#env.render()
		# 		a = no_pretrain_agent_1.choose_action(s, EPSILON)

		# 		# take action
		# 		s_, r, done, info = env.step(a)
		# 		#print(done)

		# 		# modify the reward
		# 		x, _, theta, _ = s_
		# 		r1 = (2.4 - abs(x)) / 2.4 - 0.8
		# 		r2 = (0.20944 - abs(theta)) / 0.20944 - 0.5
		# 		r_cont = r1 + r2

		# 		no_pretrain_agent_1.store_transition(s, a, r_cont, s_, done)
		# 		ep_r += r
		# 		timestep += 1
		# 		if no_pretrain_agent_1.memory_counter > config['first_update']:
		# 			no_pretrain_agent_1.learn()
		# 		if done:
		# 			print('No pretrain - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
		# 			rwd_no_pretrain_dqn.append(ep_r)
		# 			break

		# 		s = s_

		# del no_pretrain_agent_1

		# rwd_no_pretrain_dqn = utils.moving_avg(rwd_no_pretrain_dqn, winWidth)
		# tmp_rwd = np.array(rwd_no_pretrain_dqn)
		# aver_rwd_no_pretrain_dqn = aver_rwd_no_pretrain_dqn + (tmp_rwd - aver_rwd_no_pretrain_dqn)/float(exp+1)
		# if exp == 0:
		# 	var_rwd_no_pretrain = np.zeros(aver_rwd_no_pretrain_dqn.shape)
		# else:
		# 	var_rwd_no_pretrain = float(exp-1)/float(exp)*var_rwd_no_pretrain + np.square(tmp_rwd - aver_rwd_no_pretrain)/float(exp+1)


	# ======================== set memory ========================
		print('\nAgent set memory...')
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.01,
			'min_epsilon': 0.01,
			'decay_steps': 10000,
			'decay_eps':0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 50000,
			'first_update': 2000
		}
		set_memory_agent = DQN(config)
		set_memory_agent.store_batch_transitions(experiences)
		set_memory_agent.learn_step_counter = 0

		EPSILON = config['init_epsilon']

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
				a = set_memory_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				set_memory_agent.store_transition(s, a, r, s_, done)
				timestep += 1
				if set_memory_agent.memory_counter > config['first_update']:
					set_memory_agent.learn()
				if done:
					print('Set memory - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_set_memory.append(ep_r)
					break

				s = s_

		del set_memory_agent
		rwd_set_memory = utils.moving_avg(rwd_set_memory, winWidth)
		tmp_rwd = np.array(rwd_set_memory)
		pre_rwd = aver_rwd_set_memory
		aver_rwd_set_memory = aver_rwd_set_memory + (tmp_rwd - aver_rwd_set_memory)/float(exp+1)
		if exp == 0:
			var_rwd_set_memory = np.zeros(aver_rwd_set_memory.shape)
		else:
			# var_rwd_set_memory = float(exp-1)/float(exp)*var_rwd_set_memory + np.square(tmp_rwd - aver_rwd_set_memory)/float(exp+1)
			var_rwd_set_memory = var_rwd_set_memory + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_set_memory))/float(exp+1)
			var_rwd_set_memory = var_rwd_set_memory/float(exp+1)


	# ======================== only pretrain ========================
		print('\nOnly pretrain agent...')
		only_pretrain_loss = []
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.01,
			'min_epsilon': 0.01,
			'decay_steps': 10000,
			'decay_eps':0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 50000,
			'first_update': 2000
		}
		only_pretrain_agent = DQN(config)
		only_pretrain_agent.store_batch_transitions(experiences)
		for _ in range(Pretrain_steps):
			loss = only_pretrain_agent.learn()
			only_pretrain_loss.append(loss)
		only_pretrain_agent.clear_memory()
		only_pretrain_agent.learn_step_counter = 0

		EPSILON = config['init_epsilon']

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
				a = only_pretrain_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				only_pretrain_agent.store_transition(s, a, r, s_, done)
				timestep += 1
				if only_pretrain_agent.memory_counter > config['first_update']:
					loss = only_pretrain_agent.learn()
					# only_pretrain_loss.append(loss)
				if done:
					print('Only pretrain - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_only_pretrain.append(ep_r)
					break

				s = s_

		del only_pretrain_agent
		rwd_only_pretrain = utils.moving_avg(rwd_only_pretrain, winWidth)
		tmp_rwd = np.array(rwd_only_pretrain)
		pre_rwd = aver_rwd_only_pretrain
		aver_rwd_only_pretrain = aver_rwd_only_pretrain + (tmp_rwd - aver_rwd_only_pretrain)/float(exp+1)
		if exp == 0:
			var_rwd_only_pretrain = np.zeros(aver_rwd_only_pretrain.shape)
		else:
			# var_rwd_only_pretrain = float(exp-1)/float(exp)*var_rwd_only_pretrain + np.square(tmp_rwd - aver_rwd_only_pretrain)/float(exp+1)
			var_rwd_only_pretrain = var_rwd_only_pretrain + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_only_pretrain))/float(exp+1) 
			var_rwd_only_pretrain = var_rwd_only_pretrain/float(exp+1)

		if loss_verbose:
			fig, ax = plt.subplots(nrows=1, ncols=1)
			ax.plot(range(len(only_pretrain_loss)), list(only_pretrain_loss), 'b', label='only pretrain')
			ax.set_xlabel('Timesteps')
			ax.set_ylabel('Loss')
			ax.legend(loc='upper right')
			ax.grid()
			fig.savefig('./Figure/only_pretrain_loss_{}.png'.format(int(time.time())))
			plt.close(fig)


	# ======================== set memory + pretrain ========================
		print('\nPretrain agent...')
		pretrain_loss = []
		config = {
			'n_actions': N_ACTIONS,
			'n_states': N_STATES,
			'env_a_shape': ENV_A_SHAPE,
			'double_q_model': True,
			'batch_size': 128,
			'learning_rate': 5e-3,
			'init_epsilon': 0.01,
			'min_epsilon': 0.01,
			'decay_steps': 10000,
			'decay_eps':0.99,
			'discount': 0.99,
			'target_update_freq': 500,
			'memory_capacity': 50000,
			'first_update': 2000
		}
		pretrain_agent = DQN(config)
		pretrain_agent.store_batch_transitions(experiences)
		for _ in range(Pretrain_steps):
			loss = pretrain_agent.learn()
			pretrain_loss.append(loss)
		pretrain_agent.learn_step_counter = 0

		EPSILON = config['init_epsilon']

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
				a = pretrain_agent.choose_action(s, EPSILON)

				# take action
				s_, r, done, info = env.step(a)
				ep_r += r

				# modify the reward
				r = utils.modify_rwd(env_id, s_)

				pretrain_agent.store_transition(s, a, r, s_, done)
				timestep += 1
				if pretrain_agent.memory_counter > config['first_update']:
					loss = pretrain_agent.learn()
					pretrain_loss.append(loss)
				if done:
					print('Pretrain - EXP ', exp+1, '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_pretrain.append(ep_r)
					break

				s = s_

		del pretrain_agent
		rwd_pretrain = utils.moving_avg(rwd_pretrain, winWidth)
		tmp_rwd = np.array(rwd_pretrain)
		pre_rwd = aver_rwd_pretrain
		aver_rwd_pretrain = aver_rwd_pretrain + (tmp_rwd - aver_rwd_pretrain)/float(exp+1)
		if exp == 0:
			var_rwd_pretrain = np.zeros(aver_rwd_pretrain.shape)
		else:
			# var_rwd_pretrain = float(exp-1)/float(exp)*var_rwd_pretrain + np.square(tmp_rwd - pre_rwd)/float(exp+1)
			var_rwd_pretrain = var_rwd_pretrain + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_pretrain))
			var_rwd_pretrain = var_rwd_pretrain/float(exp+1)

		if loss_verbose:
			fig, ax = plt.subplots(nrows=1, ncols=1)
			ax.plot(range(len(pretrain_loss)), list(pretrain_loss), 'b', label='only pretrain')
			ax.set_xlabel('Timesteps')
			ax.set_ylabel('Loss')
			ax.legend(loc='upper right')
			ax.grid()
			fig.savefig('./Figure/only_pretrain_loss_{}.png'.format(int(time.time())))
			plt.close(fig)

	env.close()

	# Save reward plot
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_no_pretrain), 'k', label='no pre-training')
	ax.fill_between(range(MaxEpisodes), aver_rwd_no_pretrain + np.sqrt(var_rwd_no_pretrain), aver_rwd_no_pretrain - np.sqrt(var_rwd_no_pretrain), facecolor='black', alpha=0.1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_pretrain), 'b--', label='pre-training + set memory')
	ax.fill_between(range(MaxEpisodes), aver_rwd_pretrain + np.sqrt(var_rwd_pretrain), aver_rwd_pretrain - np.sqrt(var_rwd_pretrain), facecolor='blue', alpha=0.1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_only_pretrain), 'r--', label='pre-training')
	ax.fill_between(range(MaxEpisodes), aver_rwd_only_pretrain + np.sqrt(var_rwd_only_pretrain), aver_rwd_only_pretrain - np.sqrt(var_rwd_only_pretrain), facecolor='red', alpha=0.1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_set_memory), 'y--', label='set memory')
	ax.fill_between(range(MaxEpisodes), aver_rwd_set_memory + np.sqrt(var_rwd_set_memory), aver_rwd_set_memory - np.sqrt(var_rwd_set_memory), facecolor='yellow', alpha=0.1)
	ax.set_title('Number of Run: ' + str(num_of_runs))
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	ax.legend(loc='upper left')
	ax.grid()
	fig.savefig('./Figure/trial_{}.png'.format(int(time.time())))
	plt.close(fig)

