import numpy as np
from pyquaternion import Quaternion
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#this contains helper functions for parsing and visualizing the simulator's output

class Visualizer():
    def __init__(self, sampleRate = 120, fontSize = 8):
        mpl.rcParams['legend.fontsize'] = fontSize
        fig = plt.figure()
        self.ax = fig.gca(projection='3d')
        #self.ax.set_aspect('equal')
        self.objectsToVisualize = []
        self.objectsToVisualizeRotations = []
        self.objectsToVisualizeColors = [] 
        self.objectsToVisualizeLines = []
        self.objectsToVisualizeCheckpoints = []
        self.sampleRate = sampleRate

    def isolateObject(self, output, entityTracker, state_size, objectName):
        y0 = np.zeros(state_size) #the states of a single object for the duration of simulation
        firstContact = -1   #first tick at which the object is recorded
        index = 0
        i = 0
        while(i < len(entityTracker)):
            l1 = len(entityTracker[i])
            y0size = state_size * l1
        
            objPos = -1
            x = 0
            while(x < l1):
                if(entityTracker[i][x] == objectName):
                    objPos = x
                    x = l1
                x += 1
        
            if(objPos != -1):
                o = output[index+(state_size*objPos):index+(state_size*(objPos+1))]
                y0 = np.vstack((y0,o))
                if(firstContact == -1):
                    firstContact = i
            
            index = index + y0size
            i += 1
        y0 = y0[1:,:] #cut off the initialized zero row
        print("{0} began existence at tick {1} and lasted for {2} ticks".format(objectName, firstContact, y0.shape[0]))
        return y0, firstContact
        
    def trimPositions(self, samples, size):
        newPos = np.zeros(samples.shape[1])
        i = 0
        while(i < samples.shape[0]):
            if(samples[i,0] < size and samples[i,0] > -size):
                if(samples[i,1] < size and samples[i,1] > -size):
                    if(samples[i,2] < size or samples[i,2] > -size):
                        newPos = np.vstack((newPos, samples[i,:]))
            i += 1
        return newPos[1:,:]
    
    def plotObject(self, name, startTime, y0, color=0, sampleRateMod=4, rotation=False,
                     line=False, checkpoints=0, trim=False, boxsize=5):
        Xsamples = y0[0::self.sampleRate]
        if(trim):
            Xsamples = self.trimPositions(Xsamples,boxsize)
            
        colors = np.zeros((Xsamples.shape[0],3))
        i = 0
        while(i < colors.shape[0]): 
            if(color == 0):
                colors[i,0] = i / colors.shape[0]
                colors[i,1] = 0
                colors[i,2] = 0
            elif(color == 1):
                colors[i,0] = 0
                colors[i,1] = i / colors.shape[0]
                colors[i,2] = 0
            elif(color == 2):
                colors[i,0] = 0
                colors[i,1] = 0
                colors[i,2] = i / colors.shape[0]
            i += 1
        
        #plot the sampled position points with a color gradient
        #self.ax.scatter(Xsamples[:,0], Xsamples[:,1], Xsamples[:,2], 
            #label='X, every {0} ticks'.format(self.sampleRate), color=colors)
        
        if(line):
            self.ax.plot(Xsamples[:,0], Xsamples[:,1], Xsamples[:,2], color=colors[colors.shape[0]-1])
        else:
            self.ax.scatter(Xsamples[:,0], Xsamples[:,1], Xsamples[:,2], color=colors)
        
        if(checkpoints > 0): #pick out points to mark as checkpoints
            csampleRate = int(Xsamples.shape[0] / checkpoints)
            checkPointSamples = Xsamples[0::csampleRate]
            self.ax.scatter(checkPointSamples[:,0], checkPointSamples[:,1], checkPointSamples[:,2], color=(0.,0.,0.))
            
    
        if(rotation):
            Z_line = np.ones((y0.shape[0],3))
            Y_line = np.ones((y0.shape[0],3))
            X_line = np.ones((y0.shape[0],3)) 
            r = y0[:,3:7]
            i = 0
            while(i < y0.shape[0]):
                tq = Quaternion(r[i])
                trZ = tq.rotate(np.array([0.,0.,1.]))
                trY = tq.rotate(np.array([0.,1.,0.]))
                trX = tq.rotate(np.array([1.,0.,0.]))
                Z_line[i] = y0[i,0:3] + trZ
                Y_line[i] = y0[i,0:3] + trY
                X_line[i] = y0[i,0:3] + trX
                i += 1
            
            f = X_line.shape[0]
            objectStartPosition = y0[0,0:3]
            lineStartPosition = X_line[0]
            axisStart = np.vstack((objectStartPosition,lineStartPosition))
            objectEndPosition = y0[y0.shape[0]-1,0:3]
            lineEndPoint = X_line[f-1]
            axisEnd = np.vstack((objectEndPosition,lineEndPoint))

            self.ax.plot(X_line[:,0], X_line[:,1], X_line[:,2], label='X+rot*[1,0,0]   X', color=(1,0,0))
            self.ax.scatter(X_line[0,0], X_line[0,1], X_line[0,2], color='r', marker='o')                  #circle = start
            self.ax.scatter(X_line[f-1:f,0], X_line[f-1:f,1], X_line[f-1:f,2], color='r', marker='^')      #triangle = end
            self.ax.plot(axisStart[:,0],axisStart[:,1],axisStart[:,2], color = (1.,1.,.4)) 
            #line between the first position and the first X,Y,or Z marker
            self.ax.plot(axisEnd[:,0],axisEnd[:,1],axisEnd[:,2], color = 'k') 
            #line between last position and the last X,Y,or Z marker
        
            f = Y_line.shape[0]
            objectStartPosition = y0[0,0:3]
            lineStartPosition = Y_line[0]
            axisStart = np.vstack((objectStartPosition,lineStartPosition))
            objectEndPosition = y0[y0.shape[0]-1,0:3]
            lineEndPoint = Y_line[f-1]
            axisEnd = np.vstack((objectEndPosition,lineEndPoint))
        
            self.ax.plot(Y_line[:,0], Y_line[:,1], Y_line[:,2], label='X+rot*[0,1,0]   Y', color=(0,1,0))
            self.ax.scatter(Y_line[0,0], Y_line[0,1], Y_line[0,2], color='r', marker='o')                  #circle = start
            self.ax.scatter(Y_line[f-1:f,0], Y_line[f-1:f,1], Y_line[f-1:f,2], color='r', marker='^')      #triangle = end
            self.ax.plot(axisStart[:,0],axisStart[:,1],axisStart[:,2], color = (1.,1.,.4)) 
            #line between the first position and the first X,Y,or Z marker
            self.ax.plot(axisEnd[:,0],axisEnd[:,1],axisEnd[:,2], color = 'k') 
            #line between last position and the last X,Y,or Z marker
        
            f = Z_line.shape[0]
            objectStartPosition = y0[0,0:3]
            lineStartPosition = Z_line[0]
            axisStart = np.vstack((objectStartPosition,lineStartPosition))
            objectEndPosition = y0[y0.shape[0]-1,0:3]
            lineEndPoint = Z_line[f-1]
            axisEnd = np.vstack((objectEndPosition,lineEndPoint))

            self.ax.plot(Z_line[:,0], Z_line[:,1], Z_line[:,2], label='X+rot*[0,0,1]   Z', color=(0,0,1))
            self.ax.scatter(Z_line[0,0], Z_line[0,1], Z_line[0,2], color='r', marker='o')                  #circle = start
            self.ax.scatter(Z_line[f-1:f,0], Z_line[f-1:f,1], Z_line[f-1:f,2], color='r', marker='^')      #triangle = end
            self.ax.plot(axisStart[:,0],axisStart[:,1],axisStart[:,2], color = (1.,1.,.4)) 
            #line between the first position and the first X,Y,or Z marker
            self.ax.plot(axisEnd[:,0],axisEnd[:,1],axisEnd[:,2], color = 'k') 
            #line between last position and the last X,Y,or Z marker
    
    def addObjectToVisualize(self, objectName, rotation = False, color = 0, line = False, checkpoints=0):
        self.objectsToVisualize.append(objectName)
        self.objectsToVisualizeRotations.append(rotation)
        self.objectsToVisualizeColors.append(color)
        self.objectsToVisualizeLines.append(line)
        self.objectsToVisualizeCheckpoints.append(checkpoints)
        print("Visualizing {0} with color option {1}, rotation set to {2}, with line option {3}".format(objectName, color, rotation, line))
        
    def visualizeOutput(self, output, entityTracker, state_size, boxsize=10., trim=False):
        if(trim):
            #box around planet:
            self.ax.scatter(-boxsize,-boxsize,-boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter(-boxsize,-boxsize, boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter(-boxsize, boxsize,-boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter(-boxsize, boxsize, boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter( boxsize,-boxsize,-boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter( boxsize,-boxsize, boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter( boxsize, boxsize,-boxsize, marker='o', color=(0.,0.,0.))
            self.ax.scatter( boxsize, boxsize, boxsize, marker='o', color=(0.,0.,0.))
        
        self.ax.scatter(0,0,0, marker='o', color=(0.5,0.5,0.5))
                
        i = 0
        while(i < len(self.objectsToVisualize)):
            y0,start = self.isolateObject(output, entityTracker, state_size, self.objectsToVisualize[i])
            self.plotObject(self.objectsToVisualize[i], start, y0, 
                    self.objectsToVisualizeColors[i], self.sampleRate, 
                    self.objectsToVisualizeRotations[i], self.objectsToVisualizeLines[i], 
                    self.objectsToVisualizeCheckpoints[i], trim, boxsize)
            i += 1
        
        self.ax.legend()
        self.ax.set_xlabel("X (position)")
        self.ax.set_ylabel("Y (position)")
        self.ax.set_zlabel("Z (position)")
        
        plt.show()
        
        
        
        