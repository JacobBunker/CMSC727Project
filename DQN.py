import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from collections import deque

import RigidBody as rb
import Simulator as sm
import outputParse
import ActionSpace as a_s

from scipy.integrate import odeint
from pyquaternion import Quaternion

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time


#create simulation

tick_length = 1./30.  #step length between physics checks, don't change this
seconds = 50.     #seconds to simulate
step_size = 1.        # intervals to print seconds at. -1 for no print
verbose = False     #true to print

action_size = 8
state_size = 13 + action_size #don't change this, default = 13, agentActions = 8
sim = sm.Simulation(state_size, action_size) #create simulator

mass = 10. #mass of the object
dim = np.array([1.,1.,1.]) #dimensions of the cube object
x = np.array([0.,0.,5.]) #position
q = Quaternion(np.array([0.,0.,0.,1.])) #rotation
p = np.array([0.,0.,0.]) #linear momentum
l = np.array([0.,0.,0.]) #angular momentum
objectType = "Agent"
objectName = "Prime"
    #forward, back, up, down, left, right, rollleft, rollright
thrusts = np.array([1.0,0.5 ,0.5,0.5 ,0.5,0.5, 0.5,0.5]) #the thrust magnitude for the various thrusters
loadtime = 10. #ten second load time between firing projectiles
sim.createObject(mass, dim, x, q, p, l, objectType, objectName, thrusts, loadtime) #add cube object to bodies

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env # environment
        self.state_size = 13 #env.state_size # number of state parameters
        self.action_size = 2 #env.action_space.rows # number of possible actions
        self.memory = deque(maxlen=10000) # memory stores max of 10000 events
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001 # for the neural net
        self.model = self._build_model() # untrained neural net
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # Store this experience in memory
        pos = next_state[0][0]
        v = next_state[0][1]
        
        # Changing the reward function!
        #reward = abs(v) + abs(pos + 0.5)/10
        #reward = reward + 1
        
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        # Act in an epsilon greedy manner
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  
    
    def act_greedy(self, state):
        # Act in a greedy manner after environment is solved
        return np.argmax(self.model.predict(state)[0]) 
    
    def replay(self, batch_size):
        # Learn from past experiences
        if batch_size > len(self.memory):
            print("batch size greater than memory length")
            return
        minibatch = random.sample(self.memory, batch_size) # Pick a random x amount of experiences to learn from
        for state, action, reward, next_state, done in minibatch:
            target = reward 
            # If we're at a terminal state, no need to look at next state
            if not done:
                # Standard value function
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target # alpha = 1 in this agent
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
# initialize gym environment and the agent
trackTotalOutput = False #enable if you want visualization
sim.createSimulation(tick_length, trackTotalOutput)
agent = DQNAgent(sim)
episodes = 5000
rewards = deque(maxlen=100)

statesave = agent.env.reset()

#Build Memory
print("Building Memory")
for i in range(50):
    print("memory {0}".format(i))
    state = agent.env.reset()
    state = np.reshape(state[0:13], [1, agent.state_size])
    done = False
    while not done:
        action = np.array([agent.act(state)])
        next_state, reward, done = agent.env.runSimulation(action)
        next_state = np.reshape(next_state[0:13], [1, agent.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

#Learn
print("Learning")
for e in range(episodes):
    state = agent.env.reset()
    state = np.reshape(state[0:13], [1, agent.state_size])
    done = False
    R = 0
    while not done:
        action = np.array([agent.act(state)])
        next_state, reward, done = agent.env.runSimulation(action)
        next_state = np.reshape(next_state[0:13], [1, agent.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        R += reward
        if done:
            print("episode: {}/{}, reward: {}, average: {}"
                  .format(e+1, episodes, R, np.average(rewards)))
            rewards.append(R)
            break
    if e >= episodes and np.average(rewards) >= 300:
        print("Environment Solved")
        break
    agent.replay(50)
    
    
position = np.zeros(3)
    
##### VISUALIZE FINAL RUN #####
state = agent.env.reset()
state = np.reshape(state[0:13], [1, agent.state_size])
done = False
while not done:
    action = np.array([agent.act(state)])
    next_state, reward, done = env.runSimulation(action)
    position = np.vstack((position,next_state[0:3]))
    next_state = np.reshape(next_state[0:13], [1, agent.state_size])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    R += reward
    if done:
        print("reward: {0}".format(R))
        break
            
position = position[1:,:]

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
print(position)
ax.scatter(position[:,0], position[:,1], position[:,2], color=(0.,0.,0.))

ax.legend()
ax.set_xlabel("X (position)")
ax.set_ylabel("Y (position)")
ax.set_zlabel("Z (position)")

plt.show()


            
            
            
            
            
            