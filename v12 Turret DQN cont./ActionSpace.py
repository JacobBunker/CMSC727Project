import numpy as np

class ActionSpace():
    def __init__(self, size, space_type):
        self.size = size
        if(space_type == "simultaneous"):
            self.rows = np.power(2, self.size)
            self.actions = np.zeros((self.rows,self.size))
            c = 0
            t = self.rows
            while(c < size):
                t = t/2
                x = t
                i = 0
                while(i < self.rows):
                    if(x > 0):
                        self.actions[i,c] = 1
                        x -= 1
                    elif(x > -t + 1):
                        self.actions[i,c] = 0
                        x -= 1
                    else:
                        x = t
                    i += 1
                c += 1
        elif(space_type == "singular"):
            self.rows = size
            self.actions = np.zeros((self.rows,self.size))
            c = 0
            while(c < size):
                self.actions[c,c] == 1.
                c += 1
        elif(space_type == "hoverAgent"):
            self.rows = 2
            self.actions = np.zeros((self.rows, self.size))
            self.actions[0,:] = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
            self.actions[1,:] = np.array([1.,0.,0.,0.,0.,0.,0.,0.])
        elif(space_type == "turretAgent"):
            self.rows = 6
            self.actions = np.zeros((self.rows, self.size))
            self.actions[0,:] = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) #do nothing
            self.actions[1,:] = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.]) #turn up
            self.actions[2,:] = np.array([0.,0.,0.,1.,0.,0.,0.,0.,0.]) #turn down
            self.actions[3,:] = np.array([0.,0.,0.,0.,1.,0.,0.,0.,0.]) #turn left
            self.actions[4,:] = np.array([0.,0.,0.,0.,0.,1.,0.,0.,0.]) #turn right
            self.actions[5,:] = np.array([0.,0.,0.,0.,0.,0.,0.,0.,1.]) #fire projectile
            
    def sample(self):
        t = np.random.randint(self.rows)
        return t
            
            
            
            
            
            
            
            
            
            