# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import random
from collections import deque 


# Hyper Parameters:
FRAME_PER_ACTION = 4
GAMMA = 0.95 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 10000

class BrainDQN:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.highScore = 0 # MWM
		self.actions = actions # the number of actions
		self.lastAction = 1 # MWM added to support action repeat for FRAME_PER_ACTION

		# make epsilon a variable so it can be saved and restored with the model
		#self.var_timeStep = tf.Variable(self.timeStep, trainable=False, name="var_timeStep")
		self.var_epsilon = tf.Variable(0, trainable=False, name="var_epsilon")
		
		#self.var_highScore = tf.Variable(11., trainable=False, name="var_newHighScore") # MWM Added to begin tracking score
		
		# Create summary data and FileWriter
		#self.var_highScore = tf.Variable(self.highScore, trainable=False, name="var_highScore") # MWM Added to begin tracking score
		#self.summary_op = tf.summary.scalar('newHighScore', self.var_highScore)
		#self.epsilon_op = tf.summary.scalar('epsilon', self.var_epsilon)
		#self.timeStep_op = tf.summary.scalar('timeStep', self.var_timeStep)
		self.file_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())

		# init Q network - each of these is a tensor
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# init Target Q Network - each of these is a tensor
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		# this is an operation that gets run later
		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("./savedweights")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print("Successfully loaded:{}".format(checkpoint.model_checkpoint_path))
				
				# added in attemp to restore the var_epsilon
				restored_episilon = tf.get_default_graph().get_tensor_by_name("var_epsilon:0")
				#restored_timeStep = tf.get_default_graph().get_tensor_by_name("var_timeStep:0")
				print("Last saved epsilon:.{}".format(restored_episilon.eval()))
				self.epsilon = restored_episilon.eval()
				#print("Last saved timeStep:{}".format(restored_timeStep.eval()))
		else:
				print("Could not find old network weights")
	def createQNetwork(self):
		# network weights - weight_variable and bias variable methods are defined below
		W_conv1 = self.weight_variable([8,8,4,32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4,4,32,64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3,3,64,64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([3136,512])
		b_fc1 = self.bias_variable([512])

		# Note that these last two weights and bias have an output shape equal to actions
		W_fc2 = self.weight_variable([512,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# input layer
		# None allows for different batch sizes
		stateInput = tf.placeholder("float",[None,84,84,4])

		# hidden layers
		# local method conv2d takes (input, weights, stride)
		h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
		#h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
		
		# Preparing to flatten out the the last convolutional layer to feed the first
		# fully connected layer
		h_conv3_shape = h_conv3.get_shape().as_list()
		print("dimension:{}".format(h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]))
		# In the following reshape the -1 in the shape parameter has the following effect:
		# "If one component of shape is the special value -1, the size of that dimension 
		# is computed so that the total size remains constant. 
		# In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1."
		h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
		print("flattened:{}".format(h_conv3_flat))
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

		# Q Value layer
		# QValue will have a shape equal to the number of actions
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None])
		# QValue is the output layer of the Primary network.
		# actionInput will later be fed with action_batch (thus the None) from Replay
		# action_batch appears to be a one hot encoded array which would mean that 
		# the tf.multiply below results in 0 for actions not taken an QValue*1 for the
		# action taken. The subsequent reduce_sum should add up the 0's and QValue*1
		# for each observation in the batch resulting in an array of single QValues
		# for each observation. 
		# yInput will later be fed with y_batch which is calculated when the batch is 
		# taken using the recorded Reward for terminal states and Bellman's equation 
		# otherwise. Remember, the Bellman equation was defined using the Target networks
		# QValueT and not the Primary. In this way, the cost function below is reflecting
		# the difference between the Target QValueT and the Primary QValue. The optimizer 
		# will adjust the networks of the Primary network which will then manually
		# be copied to the Target Network every 10,000 steps.
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.cost_op = tf.summary.scalar('Loss', self.cost)
		self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)


	def trainQNetwork(self):

		
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		print("This is what the minibatch looks like: {}".format(minibatch))
		print("This is what action_batch looks like: {}".format(action_batch))

		# Step 2: calculate y 
		y_batch = []
		# Note that it is evaluating the output layer of the Target (QValueT) network
		# The feed dict uses the next state to support the second part of Bellman's
		# equation below
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				# np.max picks the highest QValue and adds it to the y_batch array
				# Note that the QValue_batch in the Bellman equation below is defined
				# above using the Target network and not the primary network
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
		
		# trainStep is the output of the RMSProp Optimizer
		#self.trainStep.run(feed_dict={
		#	self.yInput : y_batch,
		#	self.actionInput : action_batch,
		#	self.stateInput : state_batch
		#	})
		_, returnedCost = self.session.run([self.trainStep, self.cost],feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})

		#print("Cost after running the optimizer {}".format(returnedCost))
		loss_summary = tf.Summary(value=[tf.Summary.Value(tag="Loss", simple_value=returnedCost*100)])

		self.file_writer.add_summary(loss_summary, global_step=self.timeStep)
		self.file_writer.flush()

		# save network every 10000 
		# NEED TO PUT var_epsilon, var_timeStep, and var_score INSIDE OF AN EVAL OR RUN
		if self.timeStep % 10000 == 0:
			#self.var_timeStep = self.timeStep
			#self.var_timeStep.eval()
			#self.var_epsilon = self.epsilon
			#self.var_epsilon.eval()
			#self.var_highScore.eval()
			#summary = self.summary_op.eval()
			#self.var_highScore = self.highScore

			print("Value of self.epsilon being feed to var_epsilon {}".format(self.epsilon))
			#self.session.run(self.var_epsilon, feed_dict={self.var_epsilon: self.epsilon})
			self.var_epsilon.load(self.epsilon, self.session)
			print("Value of var_epsilon being saved:{}".format(self.var_epsilon.eval()))

			#This line saves the variables for subsequent model load, but not for TensorBoard
			self.saver.save(self.session, './savedweights/network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		
	def setPerception(self,nextObservation,action,reward,terminal):
		#print("currentState shape is {}".format(self.currentState.shape))
		#print("currentState is {}".format(self.currentState[0:2,0:2,:]))
		#print("currentState[:,:,1:] is {}".format(self.currentState[:3,:3,1:]))
		#print("nextObservation shape is {}".format(nextObservation.shape))
		#print("Next observation is {}".format(nextObservation))
		# the [:,:,1:] takes all rows, all depths/dimensions, and all but the first column
		# axis=2 means that the append will apply to columns, 1 would have applied to depth
		# and 0 to rows
		# This following line constructs the new 4 frame state by appending the current frame
		# to the previous 4 frame state minus [:,:,1:] the oldest frame
		newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		#print("newState shape after np.append is: {}".format(newState.shape))

		# Track high score MWM
		if reward > self.highScore:
			self.highScore = reward

		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"
		if self.timeStep % 10000 == 0:
			print("TIMESTEP {} / STATE {} / EPSILON {}".format(self.timeStep, state, self.epsilon))
			
			# Evaluate and write out summary highScore for TensorBoard
			#var_highScore = tf.Variable(self.highScore, trainable=False, name="var_highScore") # MWM Added to begin tracking score
			#summary_op = tf.summary.scalar('newHighScore', var_highScore)
			#summaryInit = tf.variables_initializer([self.var_highScore])
			#self.session.run(summaryInit)

			score_summary = tf.Summary(value=[tf.Summary.Value(tag="Score", simple_value=self.highScore)])			
			epsilon_summary = tf.Summary(value=[tf.Summary.Value(tag="Epsilon", simple_value=self.epsilon)])
			#loss_summary = tf.Summary(value=[tf.Summary.Value(tag="x_Loss", simple_value=self.cost.eval())])
			#loss_summary = tf.Summary(value=[tf.Summary.Value(tag="Loss", simple_value=self.cost)])

			print(self.highScore)
			#loss = self.session.run([self.cost_op])
			print('score_summary {}'.format(score_summary))
			print('epsilon_summary {}'.format(epsilon_summary))
			#print('loss {}'.format(loss))
			print('Step {}'.format(self.timeStep))

			#self.file_writer.add_summary(summary, global_step=self.timeStep)
			self.file_writer.add_summary(score_summary, global_step=self.timeStep)
			self.file_writer.add_summary(epsilon_summary, global_step=self.timeStep)
			#self.file_writer.add_summary(summary, global_step=step)
			self.file_writer.flush()

			# Reset highScore to zero
			self.highScore = 0

		# added to save epsilon
		#self.var_epsilon = self.epsilon
		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		# Look into eval and the use of interactive sessions
		# Interactive sessions are not typically used in interactive shells
		# however, it has the nice feature of allowing eval to be called on a tensor
		# without specifying the session
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		#FakeQValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})
		#print("QValue is {}".format(QValue))
		#print("QValue without [0] is {}".format(FakeQValue))

		# this sets the returned action as a one-hot vector of size number of env actions
		action = np.zeros(self.actions)
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1 # sets chosen action to 1, all others still 0
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
			self.lastAction = np.argmax(np.array(action)) # MWM lastAction should only update every 4th step
		else:
			# action[0] = 1 # do nothing
			# this essentially says don't take any further action for the 
			# duration of FRAME_PER_ACTION
			# might want to play around with taking the same ction as the last
			# 
			action_index = self.lastAction
			action[action_index] = 1 # 
			print("Last action {}".format(self.lastAction))
			print("Action onehot {}".format(action))
			# NEED TO FIGURE OUT HOW TO ACCESS PREVIOUS ACTION FROM HERE.
			# ONE POSSIBILITY IS TO ACCESS THE LAST RECORD FROM SELF.REPLAYMEMORY

		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		return action

	def setInitState(self,observation):
        # since the initial observation is only one frame and we want to stack
        # four consequtive frames, we stack the lone frame
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self,shape):
		# create a tensor of size shape with random values from a normal distribution
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride):
		# strides comes in as a scalar, but is passed to tf.nn.conv2d as
		# strides: A list of ints. 1-D tensor of length 4. The stride of the 
		# sliding window for each dimension of input. 
		# The dimension order is determined by the value of data_format, see below for details.
		# DATA_FORMAT: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". 
		# Specify the data format of the input and output data. With the default format "NHWC", 
		# the data is stored in the order of: [batch, height, width, channels]. Alternatively, 
		# the format could be "NCHW", the data storage order of: [batch, channels, height, width].
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
