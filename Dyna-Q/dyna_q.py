import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils


class Q_Net(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
		super(Q_Net, self).__init__()
		# build network layers
		self.fc1 = nn.Linear(N_STATES, H1Size)
		self.fc2 = nn.Linear(H1Size, H2Size)
		self.out = nn.Linear(H2Size, N_ACTIONS)

		# initialize layers
		utils.weights_init_normal([self.fc1, self.fc2, self.out], 0.0, 0.1)
		# utils.weights_init_xavier([self.fc1, self.fc2, self.out])

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		actions_value = self.out(x)

		return actions_value

class EnvModel(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
		super(EnvModel, self).__init__()
		# build network layers
		self.fc1 = nn.Linear(N_STATES + N_ACTIONS, H1Size)
		self.fc2 = nn.Linear(H1Size, H2Size)
		self.statePrime = nn.Linear(H2Size, N_STATES)
		self.reward = nn.Linear(H2Size, 1)
		self.done = nn.Linear(H2Size, 1)

		# initialize layers
		utils.weights_init_normal([self.fc1, self.fc2, self.statePrime, self.reward, self.done], 0.0, 0.1)
		# utils.weights_init_xavier([self.fc1, self.fc2, self.statePrime, self.reward, self.done])

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		statePrime_value = self.statePrime(x)
		reward_value = self.reward(x)
		done_value = self.done(x)
		done_value = F.sigmoid(done_value)

		return statePrime_value, reward_value, done_value

class DynaQ(object):
	def __init__(self, config):
		self.config = config
		self.n_states = self.config['n_states']
		self.n_actions = self.config['n_actions']
		self.env_a_shape = self.config['env_a_shape']
		self.Q_H1Size = 64
		self.Q_H2Size = 32
		self.env_H1Size = 64
		self.env_H2Size = 32
		self.eval_net = Q_Net(self.n_states, self.n_actions, self.Q_H1Size, self.Q_H2Size)
		self.target_net = deepcopy(self.eval_net)
		self.env_model = EnvModel(self.n_states, 1, self.env_H1Size, self.env_H2Size)
		self.learn_step_counter = 0                                     							# for target updating
		self.memory_counter = 0                                         							# for storing memory
		self.memory = np.zeros((self.config['memory_capacity'], self.n_states * 2 + 3))     		# initialize memory
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config['learning_rate'])
		self.env_opt = torch.optim.Adam(self.env_model.parameters(), lr=0.01)
		self.loss_func = nn.MSELoss()
		#self.loss_func = nn.SmoothL1Loss()

	def choose_action(self, x, EPSILON):
		x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
		# input only one sample
		if np.random.uniform() > EPSILON:   # greedy
			actions_value = self.eval_net.forward(x)
			action = torch.max(actions_value, 1)[1].data.numpy()
			action = action[0][0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  	# return the argmax index
		else:   # random
			action = np.random.randint(0, self.n_actions)
			action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
		return action

	def store_transition(self, s, a, r, s_, d):
		transition = np.hstack((s, [a, r, d], s_))
		# replace the old memory with new memory
		index = self.memory_counter % self.config['memory_capacity']
		self.memory[index, :] = transition
		self.memory_counter += 1

	def store_batch_transitions(self, experiences):
		index = self.memory_counter % self.config['memory_capacity']
		for exp in experiences:
			self.memory[index, :] = exp
			self.memory_counter += 1

	def update_env_model(self):
		sample_index = np.random.choice(min(self.config['memory_capacity'], self.memory_counter), self.config['batch_size'])
		b_memory = self.memory[sample_index, :]

		b_in = Variable(torch.FloatTensor(np.hstack((b_memory[:, :self.n_states], b_memory[:, self.n_states:self.n_states+1]))))
		b_y = Variable(torch.FloatTensor(np.hstack((b_memory[:, -self.n_states:], b_memory[:, self.n_states+1:self.n_states+2], b_memory[:, self.n_states+2:self.n_states+3]))))

		b_out = self.env_model(b_in)
		loss = self.loss_func(torch.cat(b_out, 1), b_y)
		self.env_opt.zero_grad()
		loss.backward()
		self.env_opt.step()

	def learn(self):
		# target parameter update
		if self.learn_step_counter % self.config['target_update_freq'] == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1

		# sample batch transitions
		sample_index = np.random.choice(min(self.config['memory_capacity'], self.memory_counter), self.config['batch_size'])
		b_memory = self.memory[sample_index, :]
		b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
		b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)))
		b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]))
		b_d = Variable(torch.FloatTensor(1 - b_memory[:, self.n_states+2:self.n_states+3]))
		b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

		# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
		
		if self.config['double_q_model']:
			q_eval_next = self.eval_net(b_s_)
			q_argmax = np.argmax(q_eval_next.data.numpy(), axis=1)
			q_next = self.target_net(b_s_)
			q_next_numpy = q_next.data.numpy()
			q_update = np.zeros((self.config['batch_size'], 1))
			for i in range(self.config['batch_size']):
				q_update[i] = q_next_numpy[i, q_argmax[i]]
			q_target = b_r + Variable(torch.FloatTensor(self.config['discount'] * q_update)) * b_d
		else:
			q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
			q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'], 1) * b_d  # shape (batch, 1)

		loss = self.loss_func(q_eval, q_target)
		self.optimizer.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
		# for param in self.eval_net.parameters():
		# 	param.grad.data.clamp_(-0.5, 0.5)
		self.optimizer.step()

	def simulate_learn(self):
		# target parameter update
		if self.learn_step_counter % self.config['target_update_freq'] == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1

		# sample batch transitions
		sample_index = np.random.choice(min(self.config['memory_capacity'], self.memory_counter), self.config['batch_size'])
		b_memory = self.memory[sample_index, :]
		b_s = b_memory[:, :self.n_states]

		# # cartpole random generated data
		# b_s_s = np.random.uniform(low=-2.4, high=2.4, size=(self.config['batch_size'], 1))
		# b_s_theta = np.random.uniform(low=-0.2094, high=0.2094, size=(self.config['batch_size'], 1))
		# b_s_v = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
		# b_s_w = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
		# b_s = np.hstack((b_s_s, b_s_v, b_s_theta, b_s_w))

		# mountaincar random generated data
		# b_s_s = np.random.uniform(low=-1.2, high=0.6, size=(self.config['batch_size'], 1))
		# b_s_v = np.random.uniform(low=-0.07, high=0.07, size=(self.config['batch_size'], 1))
		# b_s = np.hstack((b_s_s, b_s_v))

		b_a = np.random.randint(self.n_actions, size=b_s.shape[0])
		b_a = np.reshape(b_a, (b_a.shape[0], 1))
		b_in = Variable(torch.FloatTensor(np.hstack((b_s, np.array(b_a)))))

		statePrime_value, reward_value, done_value = self.env_model(b_in)
		b_s = Variable(torch.FloatTensor(b_s))
		b_a = Variable(torch.LongTensor(b_a))
		b_d = Variable(torch.FloatTensor(1 - done_value.data.numpy()))
		b_s_ = Variable(torch.FloatTensor(statePrime_value.data.numpy()))
		b_r = Variable(torch.FloatTensor(reward_value.data.numpy()))

		# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
		
		if self.config['double_q_model']:
			q_eval_next = self.eval_net(b_s_)
			q_argmax = np.argmax(q_eval_next.data.numpy(), axis=1)
			q_next = self.target_net(b_s_)
			q_next_numpy = q_next.data.numpy()
			q_update = np.zeros((self.config['batch_size'], 1))
			for i in range(self.config['batch_size']):
				q_update[i] = q_next_numpy[i, q_argmax[i]]
			q_target = b_r + Variable(torch.FloatTensor(self.config['discount'] * q_update)) * b_d
		else:
			q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
			q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'], 1) * b_d  # shape (batch, 1)

		loss = self.loss_func(q_eval, q_target)
		self.optimizer.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
		# for param in self.eval_net.parameters():
		# 	param.grad.data.clamp_(-0.5, 0.5)
		self.optimizer.step()
