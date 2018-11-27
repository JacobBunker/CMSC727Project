import numpy as np
from pyquaternion import Quaternion
import scipy.constants
import random
from numba import jit


@jit (nopython=True,cache=True)
def calcGravity(F, X, mass):
    planet_mass = 10000000000
    gravConstant = scipy.constants.gravitational_constant
    r = np.linalg.norm(X)
    r2 = np.power(r,3)
    return F + np.multiply(-((gravConstant)*planet_mass*mass)/r2,X)
    
@jit (nopython=True,cache=True)
def planetCollision(X, P, mass):
    planet_position = np.array([0.,0.,0.])
    planet_radius = 0.1
    
    p1 = X
    p2 = X + (P/mass)
    pA = planet_position
    
    directionP = p2 - p1
    tempB = p1 - pA
    
    t = -(np.sum(directionP*tempB))
    tempT = np.sum(directionP*directionP)
    if(tempT == 0): #divide by zero check
        return False
    t = t / tempT
    X_new = p1 + (t*directionP)
    dist1 = np.linalg.norm(X_new - p1)
    dist2 = np.linalg.norm(X_new - p2)
    dist3 = np.linalg.norm(p1 - p2)
    
    tdist = dist1 + dist2
    e = 0.0001
    if(tdist >= dist3 - e and tdist <= dist3 + e):
        AX = tempB + (t*directionP)
        minDistance = np.linalg.norm(AX)
        if(minDistance <= planet_radius):
            return True #collision = True
    return False
    
@jit (nopython=True,cache=True)
def thrustersCalculate(y, T, F, X, R, thrusts):
    goforward = y[13]
    if(goforward):
        thrust = thrusts[0]     
        forward = np.dot(R, np.array([0.,0.,1.]))     
        #forward = self.q.rotate(np.array([0.,0.,1.]))
        F = F + (forward * thrust)

    gobackward = y[14]
    if(gobackward):
        thrust = thrusts[1] 
        forward = np.dot(R, np.array([0.,0.,-1.]))
        #forward = self.q.rotate(np.array([0.,0.,-1.]))
        F = F + (forward * thrust)

    turnUp = y[15]
    if(turnUp):
        thrust = thrusts[2] 
        l1 = np.array([0.,-0.5,0.5])   #location of point one
        l2 = np.array([0.,0.5,-0.5]) 
        f1 = np.array([0.,0.,-thrust]) #force one on point one
        f2 = np.array([0.,0.,thrust])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        #f1 = self.q.rotate(f1)
        #f2 = self.q.rotate(f2)

        #rl1 = self.q.rotate(l1)
        #rl2 = self.q.rotate(l2)
        
        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)
        
        tau = rlf1 + rlf2
        T = T + tau

    turnDown = y[16]
    if(turnDown):
        thrust = thrusts[3] 
        l1 = np.array([0.,0.5,0.5])   #location of point one
        l2 = np.array([0.,-0.5,-0.5]) 
        f1 = np.array([0.,0.,-thrust]) #force one on point one
        f2 = np.array([0.,0.,thrust])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)

        tau = rlf1 + rlf2
        T = T + tau

    turnLeft = y[17]
    if(turnLeft):
        thrust = thrusts[4] 
        l1 = np.array([0.5,0.,0.5])   #location of point one
        l2 = np.array([-0.5,0.,-0.5]) 
        f1 = np.array([0.,0.,-thrust]) #force one on point one
        f2 = np.array([0.,0.,thrust])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)

        tau = rlf1 + rlf2
        T = T + tau

    turnRight = y[18]   
    if(turnRight):
        thrust = thrusts[5] 
        l1 = np.array([-0.5,0.,0.5])   #location of point one
        l2 = np.array([0.5,0.,-0.5]) 
        f1 = np.array([0.,0.,-thrust]) #force one on point one
        f2 = np.array([0.,0.,thrust])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)

        tau = rlf1 + rlf2
        T = T + tau

    rollLeft = y[19]
    if(rollLeft):
        thrust = thrusts[6] 
        l1 = np.array([0.5,0.5,0.])   #location of point one
        l2 = np.array([-0.5,-0.5,0.]) 
        f1 = np.array([0.,-thrust,0.]) #force one on point one
        f2 = np.array([0.,thrust,0.])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)

        tau = rlf1 + rlf2
        T = T + tau

    rollRight = y[20]
    if(rollRight):
        thrust = thrusts[7] 
        l1 = np.array([-0.5,0.5,0.])   #location of point one
        l2 = np.array([0.5,-0.5,0.]) 
        f1 = np.array([0.,-thrust,0.]) #force one on point one
        f2 = np.array([0.,thrust,0.])

        f1 = np.dot(R, f1)
        f2 = np.dot(R, f2)

        rl1 = np.dot(R, l1)
        rl2 = np.dot(R, l2)

        rl1m = np.array([[0.,-rl1[2],rl1[1]],[rl1[2],0.,-rl1[0]],[-rl1[1],rl1[0],0.]])
        rlf1 = np.dot(rl1m, f1)
        
        rl2m = np.array([[0.,-rl2[2],rl2[1]],[rl2[2],0.,-rl2[0]],[-rl2[1],rl2[0],0.]])
        rlf2 = np.dot(rl2m, f2)

        tau = rlf1 + rlf2
        T = T + tau
    return T,F 
    
    

class Body(): #Rigid Body 
    def __init__(self, mass, IBody, IBodyInv, X, q, P, L, objectType, objectName, thrusts, loadtime): 
        #constants
        self.mass = mass
        self.IBody = IBody
        self.IBodyInv = IBodyInv
        
        self.thrusts = thrusts
        self.loadtime = loadtime
        
        self.objectType = objectType
        self.objectName = objectName
        
        #state variables
        self.X = X
        self.q = q #the quaternion
        self.P = P
        self.L = L
        
        self.lastshot = -(loadtime + 1.) #start out being able to fire
        
        #derived quantities
        self.IInv = np.zeros([3,3])
        self.R = np.asarray([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        self.v = np.array([0.,0.,0.])
        self.omega = np.array([0.,0.,0.])
        
        #computed quantities
        self.force = np.array([0.,0.,0.])
        self.torque = np.array([0.,0.,0.])
        self.alive = True
        
    def checkFire(self, t):
        #check and see if the agent can fire
        if(t - self.lastshot > self.loadtime):
            return True
    
    def Compute_Force_and_Torque(self, y, t):
        F = np.array([0.,0.,0.])
        T = np.array([0.,0.,0.])
        
        gravity = True
        if(gravity and self.alive):   
            collision = planetCollision(self.X, self.P, self.mass)            
            if(collision):
                self.alive = False
                #print("hit planet") 
            else:
                F = calcGravity(F, self.X, self.mass)

        #state_size = 13 + 8 = 21
        if(self.objectType == "Agent" and self.alive):
            T, F = thrustersCalculate(y, T, F, self.X, self.R, self.thrusts)
            
        self.force = F
        self.torque = T
        
        

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            