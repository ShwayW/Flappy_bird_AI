# Author: Shway Wang
# Date: 2020/12/3
# Author: Shway Wang
# Date: 2020, November 1st
# Location: China Ningxia Yinchuan
import numpy as np

class Agent(object):
	def __init__(self, alpha, gamma, epsilon = 0.1):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

	def optimal_states(self, state_options):
		# state_options is a dictionary with:
		# key: state pointer
		# value: return, i.e. Q(s,a)
		ans = []
		temp = max(state_options, key = state_options.get)
		for i in state_options:
			if state_options[i] == state_options[temp]:
				ans.append(i)
		return ans

class Sarsa_Agent(Agent):
	def selectAction(self, state_options):
		# this is en epsilon-greey action selection
		# state_options is a dictionary with:
		# key: state pointer
		# value: return, i.e. Q(s,a)
		opt_option = np.random.choice(self.optimal_states(state_options))	# random tie breaking
		p = np.random.rand()	# returns random decimal between 0 and 1
		if p < self.epsilon:	# do random action
			return np.random.choice(list(state_options.keys()))
		else:	# do optimal option
			return opt_option

	def updateActionValue(self, savt, cur_state, cur_action, next_state, next_action, ret):
		# i.e Q(s,a) <- Q(s,a) + alpha[ret + gamma*Q(s',a') - Q(s,a)]
		cur_stateActionValue = savt.getStateActionValue(cur_state, cur_action)
		next_stateActionValue = savt.getStateActionValue(next_state, next_action)
		td_error = ret + self.gamma * next_stateActionValue - cur_stateActionValue
		new_value = cur_stateActionValue + self.alpha * td_error
		savt.setStateActionValue(cur_state, cur_action, new_value)

class Q_Learning_Agent(Agent):
	def selectAction(self, state_options):
		# this is en epsilon-greey action selection
		# state_options is a dictionary with:
		# key: state pointer
		# value: return, i.e. Q(s,a)
		opt_option = np.random.choice(self.optimal_states(state_options))	# random tie breaking
		p = np.random.rand()	# returns random decimal between 0 and 1
		if p < self.epsilon:	# do random action
			return np.random.choice(list(state_options.keys()))
		else:	# do optimal option
			return opt_option

	def updateActionValue(self, savt, cur_state, cur_action, opt_next_state, opt_next_action, ret):
		# i.e Q(s,a) <- Q(s,a) + alpha[ret + gamma * max(Q(s',a')) - Q(s,a)]
		cur_stateActionValue = savt.getStateActionValue(cur_state, cur_action)
		opt_next_stateActionValue = savt.getStateActionValue(opt_next_state, opt_next_action)
		td_error = ret + self.gamma * opt_next_stateActionValue - cur_stateActionValue
		new_value = cur_stateActionValue + self.alpha * td_error
		savt.setStateActionValue(cur_state, cur_action, new_value)
