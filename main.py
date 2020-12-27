"""
Author: Shway Wang
Datetime: 2020/12/8
"""
import time
from rl_agents import *
from util import *

class StateActionValueTable(object):
	def __init__(self, path):
		self.path = path
		self.content = dict()
		self.load_data()

	def addNewStateActionSet(self, cur_state, action_values):
		# action_values is a dictionary of key:action val:values
		self.content[cur_state] = action_values

	def getStateActionValue(self, state, action):
		return self.content[state][action]

	def setStateActionValue(self, state, action, value):
		self.content[state][action] = value

	def deserialize(self, str_vec):
		return tuple(str_vec.split(','))

	def load_data(self):
		with open(self.path, 'r') as f:
			content = dict()
			f_content = f.readline()
			while f_content:
				elements = f_content.split()
				state_tuple = self.deserialize(elements.pop(0))
				value = dict()
				for i in elements:
					subkey_subvalue = i.split(':')
					realSubkey = int(subkey_subvalue[0])
					value[realSubkey] = float(subkey_subvalue[1])
				content[State(value_tuple = state_tuple)] = value
				f_content = f.readline()
		self.content = content

	def safe_data(self):
		with open(self.path, 'w') as f:
			for i in self.content:
				line = i.serialize() + ' '
				for j in self.content[i]:
					line += str(j) + ':' + str(self.content[i][j]) + ' '
				f.write(line + '\n')

class State(object):
	def __init__(self, bird = None, pipe = None, value_tuple = None, state_size = 1):
		if value_tuple is not None:
			self.pipe_to_bird = int(int(value_tuple[0]) / state_size)
			self.pipetop_to_bird = int(int(value_tuple[1]) / state_size)
			self.pipebot_to_bird = int(int(value_tuple[2]) / state_size)
		else:
			self.pipe_to_bird = int((pipe.x - bird.x) / state_size)
			self.pipetop_to_bird = int((pipe.top - bird.height) / state_size)
			self.pipebot_to_bird = int((pipe.bottom - bird.height) / state_size)

	def serialize(self):
		serial = str(self.pipe_to_bird) + ','
		serial += str(self.pipetop_to_bird) + ','
		serial += str(self.pipebot_to_bird)
		return serial

	def __hash__(self):
		return hash(self.serialize())

	def __eq__(self, other):
		if not isinstance(other, State):
			return NotImplemented
		return other.serialize() == self.serialize()

	def __ne__(self, other):
		if not isinstance(other, State):
			return NotImplemented
		return other.serialize() != self.serialize()

class Game(object):
	def __init__(self, path):
		self.game_window = GameWindow()
		self.game_speed = 30
		self.bird = Bird(230,350)
		self.base = Base(FLOOR)
		self.pipes = [Pipe(700)]
		# choose to let the game play it self or play the game:
		self.auto = False
		self.train = True
		# the AI stuff:
		self.agent = Q_Learning_Agent(alpha = 1, gamma = 0.5)
		self.savt = StateActionValueTable(path) # here loads the data
		self.cur_state = None
		self.cur_action = None
		self.ret = 0
		self.next_state = None
		self.next_action = None

	def game_loop(self):
		global WIN
		win = WIN
		for _ in range(1000):
			game = True
			# Records for AI:
			episodes = 0
			for _ in range(501):
				self.bird = Bird(230,350)
				self.base = Base(FLOOR)
				self.pipes = [Pipe(700)]
				score = 0
				clock = pygame.time.Clock()
				run = True
				# one more episode:
				episodes += 1
				ret_accum = 0
				# save training result every 500 episodes:
				if episodes % 100 == 0:
					print('saving data......')
					self.savt.safe_data()
					print('data saved!')
				if self.train:
					self.agent.epsilon = 1 / episodes
				else:
					self.agent.epsilon = 0
				''' Initialize S '''
				for pipe in self.pipes:
					if not pipe.passed:
						self.cur_state = State(self.bird, pipe)
						break
				frame = 0
				frame_k = 20
				while run:
					clock.tick(self.game_speed)
					pipe_ind = 0
					# one game frame:
					self.bird.move()
					self.base.move()
					rem = []
					add_pipe = False
					for pipe in self.pipes:
						pipe.move()
						# check for collision
						if pipe.collide(self.bird, win):
							run = False
						if pipe.x + pipe.PIPE_TOP.get_width() < 0:
							rem.append(pipe)
						if not pipe.passed and pipe.x < self.bird.x:
							pipe.passed = True
							add_pipe = True
					if add_pipe:
						score += 1
						self.ret += 1000 # reward 1000 for every pipe past
						self.pipes.append(Pipe(WIN_WIDTH))
					for r in rem:
						self.pipes.remove(r)
					if self.bird.y + self.bird.img.get_height() - 10 >= FLOOR or self.bird.y < -50:
						run = False
					self.game_window.draw_window(WIN, self.bird, self.pipes, self.base, score, pipe_ind)


					# if the game is manual:
					if not self.auto:
						for event in pygame.event.get():
							if event.type == pygame.QUIT:
								game = False
								break
							if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
								self.bird.jump()
								self.cur_action = JUMP
							else:
								self.cur_action = NOTHING

		pygame.quit()
		quit()

	def agent_play(self):
		''' Choose A from S using policy derived from Q (epsilon greedy) '''
		if self.cur_state not in self.savt.content:
			if not self.auto:
				print('not in')
			action_set = {JUMP:0, NOTHING:0}
			self.savt.content[self.cur_state] = action_set
			if self.auto:
				self.cur_action = self.agent.selectAction(action_set)
		else:
			if not self.auto:
				print(self.savt.content[self.cur_state])
			else:
				self.cur_action = self.agent.selectAction(self.savt.content[self.cur_state])
		''' Take action A, observe R, S' '''
		self.ret = 0
		if self.cur_action == JUMP:
			self.bird.jump()
		# Select next action:
		for pipe in self.pipes:
			if not pipe.passed:
				self.next_state = State(self.bird, pipe)
		if self.next_state not in self.savt.content:
			action_set = {JUMP:0, NOTHING:0}
			self.savt.content[self.next_state] = action_set
			self.next_action = self.agent.selectAction(action_set)
		else:
			self.next_action = self.agent.selectAction(self.savt.content[self.next_state])
		''' Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a(Q(S',a)) - Q(S,A)] '''
		self.agent.updateActionValue(self.savt, self.cur_state, self.cur_action, self.next_state, self.next_action, self.ret)
		''' S <- S' '''
		self.cur_state = self.next_state

def main():
	Game('./rl_train_result/bird_memory.txt').game_loop()

if __name__ == '__main__':
	main()
