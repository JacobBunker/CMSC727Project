import RigidBody as rb
import Simulator as sm
import outputParse

import numpy as np
from scipy.integrate import odeint
from pyquaternion import Quaternion

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import time

state_size = 13 + 8 #don't change this, default = 13, agentActions = 8
tick_length = 1./30.  #step length between physics checks, don't change this
seconds = 50.     #seconds to simulate
step_size = 1.        # intervals to print seconds at. -1 for no print
verbose = True     #true to print


action_size = 8
state_size = 13 + action_size #don't change this, default = 13, agentActions = 8
sim = sm.Simulation(state_size, action_size) #create simulator

mass = 10. #mass of the object
dim = np.array([1.,1.,1.]) #dimensions of the cube object
x = np.array([0.,0.,1.]) #position
q = Quaternion(np.array([0.,0.,0.,1.])) #rotation
p = np.array([0.,0.,0.]) #linear momentum
l = np.array([0.,0.,0.]) #angular momentum
objectType = "Agent"
objectName = "Prime"
    #forward, back, up, down, left, right, rollleft, rollright
thrusts = np.array([50.,0.5 ,0.5,0.5 ,0.5,0.5, 0.5,0.5]) #the thrust magnitude for the various thrusters
loadtime = 10. #ten second load time between firing projectiles
sim.createObject(mass, dim, x, q, p, l, objectType, objectName, thrusts, loadtime) #add cube object to bodies
#objectName = "Second"
#x = np.array([0.,0.,10.])
#sim.createObject(mass, dim, x, q, p, l, objectType, objectName, thrusts, loadtime) #add cube object to bodies

#ATTENTION - OBJECTS MUST HAVE UNIQUE NAMES! 
#TODO: make it so that objects recieve a unique ID to fix unique name requirement

trackTotalOutput = True #enable if you want visualization

state = sim.createSimulation(tick_length, trackTotalOutput)
print("before sim:")
print(sim.yfinal)

done = False

while(done == False):
    agentNum = len(sim.bodies)
    agentActions = np.zeros(agentNum)
    i = 0
    while(i < agentNum):
        #forward, back, up, down, left, right, rollleft, rollright
        actions = 0 #np.array([0.,0., 0.,0., 0.,0., 0.,0.])
        agentActions[i] = actions
        i += 1
    state, reward, done = sim.runSimulation(agentActions)
    print(sim.yfinal)
    #a = 0
    #print(state[(a*state_size):((a+1)*state_size)])

print("after sim:")
print(sim.yfinal)
sim.reset()   
print("after reset:")
print(sim.yfinal) 
exit()

if(trackTotalOutput == False):
    exit()
    
output = sim.output
entityTracker = sim.entityTracker
#Everything past here is for visualization

maxSamples = 500
if((seconds/tick_length) < maxSamples):
    sampleRate = 1
else:
    sampleRate = int((seconds/tick_length)/maxSamples)

print(sampleRate)
    

    

    
#sampleRate = int((seconds*tick_length) * 4)


v = outputParse.Visualizer(sampleRate)
#                       Name,  Rotation, Color(0,1,2), Line?, Checkpoints
v.addObjectToVisualize("Prime", True, 0, False, -1)
#v.addObjectToVisualize("Second", False, 1, True, 10)
#v.addObjectToVisualize("Projectile_1", False, 1, True)
#v.addObjectToVisualize("Projectile_2", False, 1, True)
#v.addObjectToVisualize("Projectile_3", False, 1, True)
#v.addObjectToVisualize("Projectile_4", False, 1, True)

boxsize = 15.
trim = False
v.visualizeOutput(output, entityTracker, state_size, boxsize, trim)



