import numpy as np
import RigidBody as rb
import ActionSpace as a_s

from scipy.integrate import ode
import matplotlib.pyplot as plt

import copy

from pyquaternion import Quaternion
from numba import jit

@jit (nopython=True, cache=True)
def getIInvandOmega(R, IBodyInv, L):
    iinv = np.dot(np.dot(R,IBodyInv),np.transpose(R))
    omega = np.dot(iinv, L)
    return iinv, omega


class Simulation(): #Rigid Body 
    def __init__(self, state_size, action_size): 
        self.state_size = state_size
        self.bodies = []
        self.projectileCount = 0
        self.resetState = np.zeros(state_size)
        self.resetBodies = []
        self.action_space = a_s.ActionSpace(action_size)
    def State_To_Array(self, state):
        y = 0
        out = np.zeros(self.state_size)
    
        out[y] = state.X[0]; y+=1
        out[y] = state.X[1]; y+=1
        out[y] = state.X[2]; y+=1
    
        out[y] = state.q.real; y+=1
        t = state.q.vector
        out[y] = t[0]; y+=1
        out[y] = t[1]; y+=1
        out[y] = t[2]; y+=1
        
        out[y] = state.P[0]; y+=1
        out[y] = state.P[1]; y+=1
        out[y] = state.P[2]; y+=1
    
        out[y] = state.L[0]; y+=1
        out[y] = state.L[1]; y+=1
        out[y] = state.L[2]; y+=1
    
        return out
    

    def ddt_State_to_Array(self, state):
        y = 0
        out = np.zeros(self.state_size)
        out[y] = state.v[0]; y+=1
        out[y] = state.v[1]; y+=1
        out[y] = state.v[2]; y+=1
        
        omegaq = Quaternion(np.array(np.append([0.],state.omega)))
        qdot = (omegaq * state.q)
        qdot = 0.5 * qdot
    
        out[y] = qdot.real; y+=1
        t = qdot.vector
        out[y] = t[0]; y+=1
        out[y] = t[1]; y+=1
        out[y] = t[2]; y+=1
        
        if(state.alive):
            out[y] = state.force[0]; y+=1
            out[y] = state.force[1]; y+=1
            out[y] = state.force[2]; y+=1
        else:
            out[y] = 0; y+=1
            out[y] = 0; y+=1
            out[y] = 0; y+=1
    
        out[y] = state.torque[0]; y+=1
        out[y] = state.torque[1]; y+=1
        out[y] = state.torque[2]; y+=1
    
        return out
        
    def Array_To_State(self, state, a):
        y = 0
        state.X[0] = a[y]; y+=1
        state.X[1] = a[y]; y+=1
        state.X[2] = a[y]; y+=1
        
        r = a[y]; y+=1
        i = a[y]; y+=1
        j = a[y]; y+=1
        k = a[y]; y+=1
    
        state.q = Quaternion(np.array([r,i,j,k]))
    
        state.P[0] = a[y]; y+=1
        state.P[1] = a[y]; y+=1
        state.P[2] = a[y]; y+=1
    
        state.L[0] = a[y]; y+=1
        state.L[1] = a[y]; y+=1
        state.L[2] = a[y]; y+=1
        
        #compute the auxillary variables
    
        state.v = state.P / state.mass
        state.R = state.q.rotation_matrix
        state.IInv, state.omega = getIInvandOmega(state.R, state.IBodyInv, state.L) 
        #np.matmul(np.matmul(state.R,state.IBodyInv),np.transpose(state.R))
        #state.omega = np.matmul(state.IInv, state.L)
    
        #print(np.reshape(state.L,(3,1)))        
        return state
                    

    def Array_To_Bodies(self, a):
        newBodies = []
        i = 0
        while(i < len(self.bodies)):
            newBodies.append(self.Array_To_State(self.bodies[i],a[(i)*self.state_size:(i+1)*self.state_size]))
            i += 1
        return newBodies
    
    def Bodies_To_Array(self):
        a = np.empty(0)
        i = 0
        while(i < len(self.bodies)):
            a = np.append(a, self.State_To_Array(self.bodies[i]))
            i += 1
        return a
    
    def dydt(self, t, y):
        self.bodies = self.Array_To_Bodies(y)
        ydot = np.empty(0)
        i = 0
        while(i < len(self.bodies)):
            self.bodies[i].Compute_Force_and_Torque(y[i:i+(self.state_size)],t)
            ydot = np.append(ydot, self.ddt_State_to_Array(self.bodies[i]))
            i += 1
        return ydot
        
    def blockIBody(self,x,y,z,M,funmode=False): #dimensions x,y,z of block, mass M
        block = np.array([[(y*y)+(z*z),0.,0.],[0.,(x*x)+(z*z),0.],[0.,0.,(x*x)+(y*y)]])
        if(funmode):
            print("FUNFUNFUNFUNFUNFU")
            block = np.array([[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
        Ibody = block * (M/12)  
        return Ibody

    def createObject(self,mass,dim,X,q,P,L, objectType, objectName="unknown", thrusts=np.array([0.,0., 0.,0., 0.,0., 0.,0.]),loadtime=1e+9):
        Ibody = self.blockIBody(dim[0],dim[1],dim[2],mass)
        r = rb.Body(mass, Ibody, np.linalg.inv(Ibody), X, q, P, L, objectType, objectName, thrusts, loadtime)
        self.bodies.append(r)
    
    def addProjectile(self, parent, firespeed):
        mass = 1. #mass of the object
        dim = np.array([0.1,0.1,0.1]) #dimensions of the cube object
        x = parent.X + np.matmul(parent.R, np.array([0.,0.,0.5])) #position
        q = parent.q #rotation
        p = parent.P + np.matmul(parent.R, np.array([0.,0.,firespeed])) #linear momentum
        l = np.array([0.,0.,0.]) #angular momentum
        objectType = "Projectile"
        objectName = "Projectile_{0}".format(self.projectileCount); self.projectileCount += 1
            #forward, back, up, down, left, right, rollleft, rollright
        self.createObject(mass, dim, x, q, p, l, objectType, objectName) #add cube object to bodies
    
    def removeObject(self, y0, name):
        currentSize = y0.shape[0]
        newBodies = []
        index = 0
        i = 0
        while(i < len(self.bodies)):
            if(self.bodies[i].objectName == name):
                index = i
            else:
                newBodies.append(self.bodies[i])
            i += 1
        ynew = np.append(y0[0:(index*self.state_size)],y0[(index+1)*self.state_size:])
        self.bodies = newBodies
        return ynew        
    
    def createSimulation(self, tick_length, trackTotalOutput):
        self.trackTotalOutput = trackTotalOutput
        self.output = np.empty(0)
        self.entityTracker = []
        
        self.y0 = np.ones(self.state_size*len(self.bodies))
        self.yfinal = np.ones(self.state_size*len(self.bodies))
        self.tick_length = tick_length
        #init states -> initialize the self.bodies!
        self.t = 0.
        self.r = ode(self.dydt).set_integrator('dop853', rtol=0., atol=1e-9,nsteps=100)
        self.yfinal = self.Bodies_To_Array() 
        #self.resetState = self.yfinal
        self.resetState = copy.deepcopy(self.yfinal)
        self.resetBodies = copy.deepcopy(self.bodies)
        
    def reset(self):
        self.yfinal = copy.deepcopy(self.resetState)
        self.y0 = copy.deepcopy(self.resetState)
        self.bodies = copy.deepcopy(self.resetBodies)
        #print(self.bodies)
        self.t = 0.
        return self.yfinal
    
    def runSimulation(self, agentActions):       
        self.yfinal = self.Bodies_To_Array() 
        i = 0
        while(i < self.state_size*len(self.bodies)):
            self.y0[i] = self.yfinal[i]
            i += 1
        
        #########tracking total output##
        if(self.trackTotalOutput):
            self.output = np.append(self.output,self.y0)
            entities = []
            for b in self.bodies:
                entities.append(b.objectName)
            self.entityTracker.append(entities)
        ################################
        
        
        i = 0
        while(i < len(self.bodies)):
            if(self.bodies[i].alive == False):
                self.tempY = self.y0
                self.y0 = self.removeObject(self.y0, self.bodies[i].objectName)
                i -= 1
            i += 1
            
        if(len(self.bodies) == 0):
            #print("NO MORE BODIES!")
            print("hit planet")
            return self.tempY, self.tempR, True
        #############agent actions added here###########
        #forward, back, up, down, left, right, rollleft, rollright
        i = 0
        while(i < len(self.bodies)):
            self.y0[(i*self.state_size)+13:(i*self.state_size)+21] = self.action_space.actions[int(agentActions[i])]
            print("Thruster: {0}".format(int(agentActions[i])))
            i += 1           
            
        ###############################################
        
        self.r.set_initial_value(self.y0,0)
        self.r.integrate(self.r.t + self.tick_length)
        self.yfinal = self.r.y
        self.t += self.tick_length
        
        
        ##########################reward function#########################
        reward = 0.
        maxDistance = 10.
        targetDistance = 5.
        a = 1
        target = "Prime"
        i = 0
        while(i < len(self.bodies)):
            if(self.bodies[i].objectName == target):
                distance = np.linalg.norm(self.bodies[i].X)
                print(distance)
                temp = np.absolute(distance - targetDistance)
                reward = (-a*np.square(temp)) + 1
                self.tempR = reward
                if(distance > maxDistance):
                    print("hit maxdistance {0}".format(maxDistance))
                    return self.yfinal, reward, True
                i = len(self.bodies)
            i += 1
        ##################################################################
        return self.yfinal, reward, False








