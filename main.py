"""
Author: Shway Wang
Datetime: 2020/12/8
"""
import time
from rl_agents import *
from util import *
import tensorflow as tf
print(tf.__version__)

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
	def __init__(self, bird = None, pipe = None, value_tuple = None):
		if value_tuple is not None:
			self.pipe_to_bird = value_tuple[0]
			self.pipetop_to_bird = value_tuple[1]
			self.pipebot_to_bird = value_tuple[2]
		else:
			self.pipe_to_bird = pipe.x - bird.x
			self.pipetop_to_bird = pipe.top - bird.height
			self.pipebot_to_bird = pipe.bottom - bird.height

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

class Hypothesis(object):
	def __init__(self, value_tuple = None):
		if value_tuple is not None:
			self.W = np.array([value_tuple[0], value_tuple[1], value_tuple[2]], dtype = np.float32)
		else:
			self.W = np.array([0, 0, 0], dtype = np.float32)
		self.b = np.array([0], dtype = np.float32)
		# transform np to tf variables
		self.weight = tf.Variable(self.W, dtype = tf.float32, trainable = True)
		self.bias = tf.Variable(self.b, dtype = tf.float32, trainable = True)
		self.output_act = tf.nn.relu # activation function for the output layer

	@tf.function
	def forward(self, x):
		"""
		param x: input in the shape of a column vector
		"""
		# transform x to tensorflow object:
		h = tf.cast(x, dtype = tf.float32)
		z = tf.matmul(self.weight, h) + self.bias
		o = self.output_act(z)
		return o

	@tf.function
	def measure_loss(self, target, prediction):
		"""
		param target: array of target values
		param prediction: array of predictions
		"""
		target = tf.cast(target, dtype = tf.float32)
		return tf.reduce_mean(tf.pow(tf.subtract(target, prediction), 2))

	@tf.function
	def compute_gradient(self, observation, target):
		"""
		param observation: input to the network
		param target: target value
		"""
		with tf.GradientTape() as tape:
			pred = self.forward(observation)
			error = self.measure_loss(target, pred)
			gradients = tape.gradient(error, self.weight + self.bias)
		return gradients # gradients[0] is dError_dW and gradients[1] is dError_db

	@tf.function
	def apply_gradient(self, ws, bs, ws_grads, bs_grads, stepsize = 0.001):
		"""
		param ws: weights of the network
		param bs: bias of the network
		param ws_grads: gradients of the weights of the network
		param bs_grads: gradients of the biases of the network
		param stepsize: stepsize parameter of the GD
		"""
		for i in range(len(ws)):
			assert isinstance(ws[i], tf.Variable)
			ws[i].assign_add(-stepsize * ws_grads[i])
		for i in range(len(bs)):
			assert isinstance(bs[i], tf.Variable)
			bs[i].assign_add(-stepsize * bs_grads[i])


class Game(object):
	def __init__(self, path):
		self.game_window = GameWindow()
		self.game_speed = 30
		# choose to let the game play it self or play the game:
		self.auto = False
		self.train = False
		# the SL stuff:
		self.hypothesis = Hypothesis()

	def game_loop(self):
		global WIN
		win = WIN
		for _ in range(1000):
			game = True
			# Records for AI:
			episodes = 0
			# train for k episodes:
			k = 501
			for _ in range(k):
				bird = Bird(230,350) # bird initial spot
				base = Base(FLOOR)
				pipes = [Pipe(700)] # pipe initial spot
				score = 0
				clock = pygame.time.Clock()
				run = True
				''' Initialize S '''
				for pipe in pipes:
					if not pipe.passed:
						self.cur_state = State(bird, pipe)
						break
				while run:
					run, score = self.episode(clock, bird, base, pipes, win, run, score)
		pygame.quit()
		quit()

	def episode(self, clock, bird, base, pipes, win, run, score):
		clock.tick(self.game_speed)
		pipe_ind = 0
		bird.move()
		if not self.auto: # if the game mode is set to manual:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					game = False
					break
				if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
					bird.jump()
					self.cur_action = JUMP
				else:
					self.cur_action = NOTHING

		if self.cur_action == JUMP:
			bird.jump()
		base.move()

		rem = []
		add_pipe = False
		for pipe in pipes:
			pipe.move()
			# check for collision
			if pipe.collide(bird, win):
				run = False
				break
			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)
			if not pipe.passed and pipe.x < bird.x:
				pipe.passed = True
				add_pipe = True
		if add_pipe:
			score += 1
			pipes.append(Pipe(WIN_WIDTH))
		for r in rem:
			pipes.remove(r)
		if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
			run = False
		if run == False:

			print("here")
		self.game_window.draw_window(WIN, bird, pipes, base, score, pipe_ind)
		return (run, score)

def main():
	Game('./rl_train_result/bird_memory.txt').game_loop()

if __name__ == '__main__':
	main()
